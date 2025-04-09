import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseArray
from rclpy.node import Node

from .utils import LineTrajectory


class PurePursuit(Node):
    """ Implements Pure Pursuit trajectory tracking with a fixed lookahead and speed.
    """

    def __init__(self):
        super().__init__("trajectory_follower")
        # Topics to be used.
        self.declare_parameter('odom_topic', "default")
        self.declare_parameter('drive_topic', "default")

        self.odom_topic: str = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.drive_topic: str = self.get_parameter('drive_topic').get_parameter_value().string_value

        # Pure Pursuit parameters.
        self.declare_parameter('lookahead', 0.5)
        self.declare_parameter('speed', 0.5)
        self.declare_parameter('wheelbase_length', 0.3302)

        self.lookahead: float = self.get_parameter('lookahead').get_parameter_value().double_value
        self.speed: float = self.get_parameter('speed').get_parameter_value().double_value
        self.wheelbase_length: float = self.get_parameter('wheelbase_length').get_parameter_value().double_value

        # Subscribers to the planned path and publishers for the drive command.
        self.trajectory = LineTrajectory("/followed_trajectory")

        self.traj_sub = self.create_subscription(
            PoseArray, "/trajectory/current",
            self.trajectory_callback, 1
        )
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, self.drive_topic, 1
        )

    def pose_callback(self, odometry_msg):
        raise NotImplementedError

    def trajectory_callback(self, msg):
        self.get_logger().info(f"Receiving new trajectory {len(msg.poses)} points")

        self.trajectory.clear()
        self.trajectory.fromPoseArray(msg)
        self.trajectory.publish_viz(duration=0.0)

        self.initialized_traj = True


def main(args=None):
    rclpy.init(args=args)
    follower = PurePursuit()
    rclpy.spin(follower)
    rclpy.shutdown()
