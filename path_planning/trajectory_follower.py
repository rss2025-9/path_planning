import rclpy
from rclpy.node import Node
# Driving.
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
# Geometry.
from geometry_msgs.msg import Pose, PoseArray
from tf_transformations import euler_from_quaternion

import numpy as np
import numpy.typing as npt

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
        self.declare_parameter('speed', 1.0)
        self.declare_parameter('wheelbase_length', 0.3302)
        self.declare_parameter('interpolation_iterations', 10)

        self.lookahead: float = self.get_parameter('lookahead').get_parameter_value().double_value
        self.speed: float = self.get_parameter('speed').get_parameter_value().double_value
        self.wheelbase_length: float = self.get_parameter('wheelbase_length').get_parameter_value().double_value
        self.interpolation_iterations: int = self.get_parameter('interpolation_iterations').get_parameter_value().integer_value

        # Subscribers to the planned path and publishers for the drive command.
        self.trajectory: LineTrajectory = LineTrajectory("/followed_trajectory")
        self.initialized_traj = False

        self.traj_sub = self.create_subscription(
            PoseArray, "/trajectory/current",
            self.trajectory_callback, 1
        )
        self.odom_sub = self.create_subscription(
            Odometry, self.odom_topic,
            self.pose_callback, 1
        )
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, self.drive_topic, 1
        )

    def pose_callback(self, odometry_msg: Odometry):
        """
        Takes the current position of the robot, finds the nearest point on the
        path, sets that as the goal point, and navigates towards it.
        """
        # Gets vectorized Pose of the robot.
        pose: Pose = odometry_msg.pose.pose
        position: npt.NDArray = np.array([pose.position.x, pose.position.y])
        yaw: np.float64 = -euler_from_quaternion(
            [pose.orientation.x, pose.orientation.y,
             pose.orientation.z, pose.orientation.w]
        )[2]

        # Moves only if the trajectory is initialized, otherwise publish stop.
        if not self.initialized_traj:
            self.get_logger().warning("Trajectory not initialized yet.")
            # Publish a stop command.
            drive_cmd: AckermannDriveStamped = AckermannDriveStamped()
            drive_cmd.drive.speed = 0.0
            drive_cmd.drive.steering_angle = 0.0
            drive_cmd.drive.steering_angle_velocity = 0.0
            self.drive_pub.publish(drive_cmd)
            return
        
        # Calculates the distance to all points in the trajectory, vectorized.
        trajectory_points: npt.NDArray[np.float64] = np.array(
            self.trajectory.points
        )
        relative_positions: npt.NDArray = trajectory_points - position  # vector from vehicle to each trajectory point
        # Rotates relative positions to the vehicle's frame.
        rotation_matrix: npt.NDArray = np.array([
            [np.cos(yaw), -np.sin(yaw)],
            [np.sin(yaw), np.cos(yaw)]
        ])
        relative_positions = relative_positions @ rotation_matrix.T
        # Calculates the distances to each point.
        distances: npt.NDArray = np.linalg.norm(relative_positions, axis=1)
        # Replaces the distance of behind points with a large value.
        distances_ahead: npt.NDArray = np.where(
            relative_positions[:, 0] > 0,
            distances,
            np.inf
        )
        # Finds the index of the closest point ahead.
        closest_idx = np.argmin(distances_ahead)

        # Only consider points with positive projection on the heading vector.
        goal_point: npt.NDArray[np.float64] = None
        if not np.any(relative_positions[:, 0] > 0):
            # If no points are ahead, default to the closest point.
            goal_point = relative_positions[np.argmin(distances)]
        elif distances[closest_idx] >= self.lookahead:
            # If the closest point is beyond the lookahead distance, use it.
            goal_point = relative_positions[closest_idx]
        else:
            # Find the first point that is within the lookahead distance.
            prev_point = trajectory_points[closest_idx]
            for i in range(closest_idx + 1, len(trajectory_points)):
                if distances[i] >= self.lookahead:
                    # Linearly interpolate to find the goal point.
                    low_point = prev_point
                    mid_point = trajectory_points[i]
                    for _ in range(self.interpolation_iterations):
                        midpoint = (low_point + mid_point) / 2
                        self.get_logger().info(f"Midpoint: {midpoint}")
                        if np.isclose(np.linalg.norm(midpoint - position), self.lookahead, atol=0.01):
                            # Converts midpoint to the vehicle's frame.
                            goal_point = (midpoint - position) @ rotation_matrix.T
                            break
                    break
                prev_point = trajectory_points[i]
            # If no points are found, use the last point.
            if goal_point is None:
                goal_point = relative_positions[-1]

        # Fallback if no suitable goal point was found.
        if goal_point is None:
            goal_point = relative_positions[-1]
        self.get_logger().info(f"Goal point: {goal_point}")

        # Calculate the curvature 
        gamma: float = 2 * goal_point[1] / (self.lookahead ** 2)
        # Calculate the steering angle.
        steering_angle: float = np.arctan(gamma * self.wheelbase_length)
        self.get_logger().info(f"Steering angle: {steering_angle}")
        
        # Create the drive command.
        drive_cmd: AckermannDriveStamped = AckermannDriveStamped()
        drive_cmd.drive.speed = self.speed
        drive_cmd.drive.steering_angle = steering_angle
        drive_cmd.drive.steering_angle_velocity = 0.0
        # Publish the drive command.
        self.drive_pub.publish(drive_cmd)

    def trajectory_callback(self, msg):
        self.get_logger().info(f"Receiving new trajectory {len(msg.poses)} points")

        self.trajectory.clear()
        # Converts from poses to the utility trajectory class.
        self.trajectory.fromPoseArray(msg)
        self.trajectory.publish_viz(duration=0.0)
        # flag to check that we have a trajectory.
        self.initialized_traj = True


def main(args=None):
    rclpy.init(args=args)
    follower = PurePursuit()
    rclpy.spin(follower)
    rclpy.shutdown()
