import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Odometry, Pose, PoseArray
from rclpy.node import Node

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
        self.declare_parameter('lookahead', 3.0)
        self.declare_parameter('speed', 1.0)
        self.declare_parameter('wheelbase_length', 0.3302)

        self.lookahead: float = self.get_parameter('lookahead').get_parameter_value().double_value
        self.speed: float = self.get_parameter('speed').get_parameter_value().double_value
        self.wheelbase_length: float = self.get_parameter('wheelbase_length').get_parameter_value().double_value

        # Subscribers to the planned path and publishers for the drive command.
        self.trajectory: LineTrajectory = LineTrajectory("/followed_trajectory")

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
        heading: np.float64 = 2 * np.arccos2(pose.orientation.w)

        # Finds the path point closest to the robot.
        if not self.initialized_traj:
            return
        
        # Calculates the distance to all points in the trajectory, vectorized.
        trajectory_points: npt.NDArray[np.float64] = np.array(
            self.trajectory.points
        )
        distances: npt.NDArray[np.float64] = np.linalg.norm(
            trajectory_points - position, axis=1
        )

        # Finds the index of the closest point.
        closest_index: int = np.argmin(distances)
        closest_point: npt.NDArray[np.float64] = trajectory_points[closest_index]

        # Finds the lookahead/goal point.
        for i in range(closest_index, len(trajectory_points)):
            # If the distance is equal, set it as the goal point.
            if distances[i] == self.lookahead:
                goal_point: npt.NDArray[np.float64] = trajectory_points[i]
                break
            # If not, interpolate between the two points.
            elif distances[i] > self.lookahead:
                goal_point: npt.NDArray[np.float64] = (
                    trajectory_points[i - 1] +
                    (self.lookahead - distances[i - 1]) /
                    (distances[i] - distances[i - 1]) *
                    (trajectory_points[i] - trajectory_points[i - 1])
                )
                break
        
        # Transform the goal point to the robot's frame.
        goal_point = np.array([
            goal_point[0] - position[0],
            goal_point[1] - position[1]
        ])
        # Rotate the goal point by the robot's heading.
        goal_point = np.array([
            goal_point[0] * np.cos(heading) + goal_point[1] * np.sin(heading),
            -goal_point[0] * np.sin(heading) + goal_point[1] * np.cos(heading)
        ])

        # Calculate the curvature 
        gamma: float = 2 * goal_point[0] / (self.lookahead ** 2)
        # Calculate the steering angle.
        steering_angle: float = np.arctan(gamma * self.wheelbase_length)
        
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
