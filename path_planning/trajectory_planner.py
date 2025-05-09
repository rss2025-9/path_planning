#QUESTIONS:
# 1) line 132 in is_free do we return cell >=0 and cell<0.65
# 2) why are points not on path 0, -1
# 3) why are we not seeing any real values for points (always 0 or -1)  

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PoseArray, Pose
from nav_msgs.msg import OccupancyGrid
from .utils import LineTrajectory

import numpy as np
import math
import random



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

        ##
        self.map = None
        self.resolution = None
        self.origin = None
        self.start = None
        self.goal = None
        ##

        # Transformation matrix to convert between map and world
        x = 25.900000
        y = 48.500000
        theta = 3.14
        self.map_resolution = 0.0504  # make sure this matches the actual map
        self.transform = np.array([
            [np.cos(theta), -np.sin(theta), x],
            [np.sin(theta),  np.cos(theta), y],
            [0,              0,             1]
        ])


    ## given callback funcs implemented
    def map_cb(self, msg):
        """Stores occupancy grid and converts unknowns (-1) to occupied (1)"""
        raw_map = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        self.map = (raw_map == -1).astype(int)  #replace -1 with 1 (occupied)
        self.map_resolution = msg.info.resolution
        self.map_origin = msg.info.origin.position
        self.get_logger().info("Map received")


    def goal_cb(self, msg):
        """callback for goal pose. calls the planning to start """
        self.goal_pose=msg.pose
        self.get_logger().info("Goal pose received")
        if self.start_pose and self.map is not None: #calls path plan
            self.plan_path(self.start_pose, self.goal_pose)
        else:
            self.get_logger().warn("Waiting for start pose or map")





    def pose_cb(self, pose):
        """callback for initial pose estimate """
        self.start_pose=pose.pose.pose
        self.get_logger().info("Start pose received")


    ######## my functions below here

    def w_to_m(self, x, y):
        """World to map index using transform matrix (inverse of the transform)."""
        point = np.array([x, y, 1.0])
        pixel = self.transform @ point
        pixel = pixel / self.map_resolution
        return int(pixel[1]), int(pixel[0])  # (row, col)

    def map_to_world(self, col, row):
        """Map index to world coordinates using inverse transform."""
        pixel = np.array([col * self.map_resolution, row * self.map_resolution, 1.0])
        point = np.linalg.inv(self.transform) @ pixel
        return float(point[0]), float(point[1])



    def is_free(self, x, y):
        """returns True if world coordinate (x, y) is in free or unknown space (but not occupied)."""
        map_y, map_x = self.w_to_m(x, y)
        if 0 <= map_x < self.map.shape[1] and 0 <= map_y < self.map.shape[0]: #checks that the map indices are within bounds of the occupancy grid
            cell=self.map[map_y, map_x] #cell is 0 if free, 1 if occupied
            return cell==0  #return 1
        return False #return 0

    

    def dist(self, pt_1, pt_2):
        return math.hypot(pt_1[0]-pt_2[0], pt_1[1]-pt_2[1])

    def rand_free(self):
        """Sample a random free point in the map and return it in world coordinates."""
        while True:
            row = random.randint(0, self.map.shape[0] - 1)
            col = random.randint(0, self.map.shape[1] - 1)

            #if self.map[row, col] >= 0 and self.map[row, col] < 0.65:
            if self.map[row, col]==0:
                x, y = self.map_to_world(col, row)
                return x, y


    def steer(self, start, end, step=0.5):
        """ steer from start to end w/ given step """
        ang=math.atan2(end[1]-start[1], end[0]-start[0])

        updated_x=start[0]+step*math.cos(ang)
        updated_y=start[1]+step* math.sin(ang)

        return updated_x, updated_y



    ####################
    

    def plan_path(self, start, end):
        """ RRT """

        #x,y, coords from pose msgs
        start=(start.position.x, start.position.y)
        goal=(end.position.x, end.position.y)

        self.get_logger().info(f"Start: {start}, Goal: {goal}")

        goal_mx, goal_my = self.w_to_m(goal[0], goal[1])
        self.get_logger().info(f"goal_mx, goal_mx ({goal_mx}, {goal_my})")
       
        if 0 <= goal_mx < self.map.shape[1] and 0 <= goal_my < self.map.shape[0]:
            cell_value=self.map[goal_my, goal_mx]
            self.get_logger().info(f"Occupancy at goal cell ({goal_mx}, {goal_my}) = {cell_value}")
        else:
            self.get_logger().warn("Goal is out of map bounds!")


        #tree init
        nodes=[start]
        parents={start: None} #keep parents for when we go backwards

        #CHANGE
        tol=0.3 #tolerance of being clsoe enough to goal
        max_iter=5000

        for _ in range(max_iter):
            rand_free_pt=self.rand_free() #start w random free point
            closest=min(nodes, key=lambda n: self.dist(n, rand_free_pt)) #nearest node already in tree to rand_free
            new_node=self.steer(closest, rand_free_pt) #steer from clsoest node to free point and rcord new node


            if self.is_free(*new_node): #add node only if its free
                #
                self.get_logger().info("new node free")
                newnode_my, newnode_mx = self.w_to_m(new_node[0], new_node[1])
                cell_value=self.map[newnode_my, newnode_mx]
                self.get_logger().info(f"Occupancy at new node ({newnode_mx}, {newnode_mx}) = {cell_value}") #should be -1, or 0 to 1
                #
                nodes.append(new_node)
                #self.get_logger().info(f"Nodes: {nodes}")
                parents[new_node]=closest

                if self.dist(new_node, goal) < tol and self.is_free(*goal): #if close enough to stop and goal free
                    parents[goal] = new_node
                    self.get_logger().info("Goal reached")
                    break

        if goal not in parents:
            # Didn't reach goal — try closest node
            closest_to_goal = min(nodes, key=lambda n: self.dist(n, goal))
            self.get_logger().warn("Goal not reached — using closest node instead.")
            now=closest_to_goal
        else:
            now=goal
            self.get_logger().info(f"Planned {now} waypoints")

        #go backwards from goal (or closest) to start to get path
        path=[]


        while now is not None:
            path.append(now)
            now=parents.get(now)
        path.reverse() #reverse to have start to goal path

        self.get_logger().info(f"Planned {len(path)} waypoints")


        #write them as poses
        self.trajectory.clear()
        for point in path:
            self.trajectory.addPoint(point)


        #publish it
        self.traj_pub.publish(self.trajectory.toPoseArray())
        self.trajectory.publish_viz()
        ##



def main(args=None):
    rclpy.init(args=args)
    planner = PathPlan()
    rclpy.spin(planner)
    rclpy.shutdown()
