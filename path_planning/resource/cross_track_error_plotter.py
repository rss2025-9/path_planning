import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseArray
from std_msgs.msg import Float32
import numpy as np
import math

def quaternion_to_yaw(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class CrossTrackErrorNode(Node):
    def __init__(self):
        super().__init__('cross_track_error_node')

        # parameters
        self.declare_parameter('path_topic', '/trajectory/current')
        self.declare_parameter('odom_topic', '/pf/pose/odom')
        self.declare_parameter('error_topic', '/cross_track_error')

        path_topic  = self.get_parameter('path_topic').get_parameter_value().string_value
        odom_topic  = self.get_parameter('odom_topic').get_parameter_value().string_value
        error_topic = self.get_parameter('error_topic').get_parameter_value().string_value

        # subs & pubs
        self.path_sub = self.create_subscription(PoseArray, path_topic, self.path_callback, 10)
        self.odom_sub = self.create_subscription(Odometry,  odom_topic, self.odom_callback, 10)
        self.error_pub = self.create_publisher(Float32, error_topic, 10)

        # storage for path & errors
        self.path_points = []      # [(x,y), …]
        self.cte_values  = []      # [cte, …]
        self.cte_times   = []      # [sec_since_start, …]
        self.start_time  = self.get_clock().now().nanoseconds * 1e-9

        self.get_logger().info('Cross-track error node initialized.')

    def path_callback(self, msg: PoseArray):
        self.path_points = [(p.position.x, p.position.y) for p in msg.poses]

    def odom_callback(self, msg: Odometry):
        if not self.path_points:
            return

        x   = msg.pose.pose.position.x
        y   = msg.pose.pose.position.y
        yaw = quaternion_to_yaw(msg.pose.pose.orientation)

        cte = self.compute_cross_track_error((x, y), yaw, self.path_points)

        # record timestamp and error
        now = self.get_clock().now().nanoseconds * 1e-9 - self.start_time
        self.cte_times.append(now)
        self.cte_values.append(cte)

        # publish
        m = Float32()
        m.data = float(cte)
        self.error_pub.publish(m)

    def compute_cross_track_error(self, position, yaw, path_points):
        p = np.array(position)
        min_dist = float('inf')
        signed_error = 0.0

        for i in range(len(path_points) - 1):
            p1 = np.array(path_points[i])
            p2 = np.array(path_points[i + 1])
            v  = p2 - p1
            w  = p  - p1

            seg_len_sq = v.dot(v)
            if seg_len_sq == 0:
                continue
            t = w.dot(v) / seg_len_sq
            closest = p1 if t < 0 else (p2 if t > 1 else (p1 + t * v))
            vec   = p - closest
            dist  = np.linalg.norm(vec)
            cross_z = v[0] * vec[1] - v[1] * vec[0]
            current_signed = np.sign(cross_z) * dist

            if dist < min_dist:
                min_dist    = dist
                signed_error = current_signed

        return signed_error

    def plot_errors(self):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print('matplotlib not available, cannot plot.')
            return

        if not self.cte_times:
            print('No CTE data collected; skipping plot.')
            return

        plt.figure()
        plt.plot(self.cte_times, self.cte_values)
        plt.xlabel('Time (s since start)')
        plt.ylabel('Cross-track error (m)')
        plt.title('Cross-track Error Over Time')
        plt.grid(True)
        plt.tight_layout()

        # save and optionally show
        plt.savefig('cross_track_error.png')
        print('Saved plot to cross_track_error.png')
        try:
            plt.show()
        except Exception:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = CrossTrackErrorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # use print to avoid logger errors after shutdown
        print('Shutting down, generating CTE plot…')
        node.plot_errors()
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception as e:
            print(f'rclpy.shutdown() error: {e}')


if __name__ == '__main__':
    main()
