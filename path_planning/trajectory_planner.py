import rclpy
from rclpy.node import Node

assert rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PoseArray
from nav_msgs.msg import OccupancyGrid
from .utils import LineTrajectory

import numpy as np
import heapq
import math


class PathPlan(Node):
    """ Listens for goal pose published by RViz and uses it to plan a path from
    current car pose.
    """

    def __init__(self):
        super().__init__("trajectory_planner")
        self.declare_parameter('odom_topic', "default")
        self.declare_parameter('map_topic', "default")
        self.declare_parameter('initial_pose_topic', "default")

        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.map_topic = self.get_parameter('map_topic').get_parameter_value().string_value
        self.initial_pose_topic = self.get_parameter('initial_pose_topic').get_parameter_value().string_value

        self.map_sub = self.create_subscription(
            OccupancyGrid,
            self.map_topic,
            self.map_cb,
            1)

        self.goal_sub = self.create_subscription(
            PoseStamped,
            "/goal_pose",
            self.goal_cb,
            10
        )

        self.traj_pub = self.create_publisher(
            PoseArray,
            "/trajectory/current",
            10
        )

        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            self.initial_pose_topic,
            self.pose_cb,
            10
        )

        self.trajectory = LineTrajectory(node=self, viz_namespace="/planned_trajectory")

    def map_cb(self, msg: OccupancyGrid):
        """Takes the Occupancy Grid of the map and creates an internal representation"""
        safety_threshold = 50

        map_width = msg.info.width
        map_height = msg.info.height
        map_data = np.array(msg.data).reshape((map_height, map_width))

        # Mark the grid as 1 if its occupancy probability is greater than safety threshold
        self.map = (map_data >= safety_threshold).astype(int)


    def pose_cb(self, pose: PoseWithCovarianceStamped):
        """Sets initial pose"""
        self.initial_pose = pose


    def goal_cb(self, msg: PoseStamped):
        """Sets goal pose"""
        self.goal_pose = msg
        if self.initial_pose is None:
            self.get_logger().warn("Initial pose is not set!")
            return
        if self.map is None:
            self.get_logger().warn("Map not found!")
            return
        
        self.plan_path(self.initial_pose, self.goal_pose, self.map)

    def plan_path(self, start_point, end_point, map):
        self.traj_pub.publish(self.trajectory.toPoseArray())
        self.trajectory.publish_viz()

    def a_star():
        pass

    def heuristic(a, b):
        # Euclidean distance from a to b
        x = abs(a[0] - b[0])
        y = abs(a[1] - b[1])
        return math.hypot(x, y)
    
    def get_neighbors()

def main(args=None):
    rclpy.init(args=args)
    planner = PathPlan()
    rclpy.spin(planner)
    rclpy.shutdown()
