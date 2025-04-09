import rclpy
from rclpy.node import Node

assert rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PoseArray
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


    ## given callback funcs implemented
    def map_cb(self, msg):
        """stores occupancy grid  """
        self.map=np.array(msg.data).reshape((msg.info.height, msg.info.width))
        self.map_resolution=msg.info.resolution
        self.map_origin=msg.info.origin.position


    def goal_cb(self, msg):
        """callback for goal pose. calls the planning to start """
        self.goal_pose = msg.pose

        if self.start_pose and self.map is not None: #calls path plan
            self.plan_path(self.start_pose, self.goal_pose)




    def pose_cb(self, pose):
        """callback for initial pose estimate """
        self.start_pose=pose.pose.pose

    ######## my functions below here

    def w_to_m(self, x, y):
        """ world coordinates to map coordinates"""
        return (int((x-self.map_origin.x)/self.map_resolution)), (int((y-self.map_origin.y)/self.map_resolution))


    def is_free(self, x, y):
        """see if point x,y in map is free"""
        map_x, map_y=self.w_to_m(x, y)

        if 0<=map_x<self.map.shape[1] and 0<=map_y<self.map.shape[0]:
            return self.map[map_y, map_x]<50  #-1=unknown, 0=free, 100=not free

        return False

    def dist(self, pt_1, pt_2):
        return math.hypot(pt_1[0]-pt_2[0], pt_1[1]-pt_2[1])

    def rand_free(self):
        """ get rand free point on map"""
        while True:
            x=random.uniform(self.map_origin.x, self.map_origin.x+self.map.shape[1]*self.map_resolution)
            y=random.uniform(self.map_origin.y, self.map_origin.y+self.map.shape[0]*self.map_resolution)
            
            if self.is_free(x, y):
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

        #tree init
        nodes=[start]
        parents={start: None} #keep parents for when we go backwards

        #CHANGE
        tol=0.3 #tolerance of being clsoe enough to goal
        max_iter=2000

        for _ in range(max_iter):
            rand_free_pt=self.rand_free() #start w random free point
            closest=min(nodes, key=lambda n: self.dist(n, rand_free_pt)) #nearest node already in tree to rand_free
            new_node=self.steer(closest, rand_free_pt) #steer from clsoest node to free point and rcord new node


            if self.is_free(*new_node): #add node only if its free
                nodes.append(new_node)
                parents[new_node]=closest

                if self.dist(new_node, goal) < tol: #if close enough stop
                    parents[goal]=new_node
                    break

        #go backwards from goal to start to get path
        path=[]
        now=goal

        while now is not None:
            path.append(now)
            now=parents.get(now)
        path.reverse() #reverse to have start to goal path


        #write them as poses
        self.trajectory.clear()
        for x, y in path:
            pose=Pose()
            pose.position.x=float(x)
            pose.position.y=float(y)
            self.trajectory.addPose(pose)

        #publish it
        self.traj_pub.publish(self.trajectory.toPoseArray())
        self.trajectory.publish_viz()
        ##



def main(args=None):
    rclpy.init(args=args)
    planner = PathPlan()
    rclpy.spin(planner)
    rclpy.shutdown()
