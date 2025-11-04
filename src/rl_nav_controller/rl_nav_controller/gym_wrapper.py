import rclpy
from rclpy.node import Node
import gym
from gym import spaces
import numpy as np
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import math
import time
import random  # 修复 random 未导入的问题

class RLNavEnv(gym.Env):
    """
    自定义强化学习环境（PPO训练局部路径控制器）
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(RLNavEnv, self).__init__()

        # ROS2 初始化
        rclpy.init(args=None)
        self.node = Node("rl_env_node")

        # 发布速度指令
        self.cmd_pub = self.node.create_publisher(Twist, '/cmd_vel', 10)
        # 发布随机目标点
        self.goal_pub = self.node.create_publisher(PoseStamped, '/goal_pose', 10)

        # 订阅激光和里程计
        self.scan = None
        self.odom = None
        self.node.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.node.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        # 环境参数
        self.max_steps = 500
        self.step_count = 0
        self.goal = np.array([1.0, 1.0], dtype=np.float32)  # 初始目标

        # 状态与动作空间
        self.laser_dim = 180
        self.state_dim = self.laser_dim + 4  # laser + vx + vz + distance + angle
        self.action_space = spaces.Box(low=np.array([0.0, -1.0]), high=np.array([0.3, 1.0]), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)

    def scan_callback(self, msg):
        self.scan = np.array(msg.ranges[:self.laser_dim])
        self.scan[np.isinf(self.scan)] = 8.0

    def odom_callback(self, msg):
        self.odom = msg

    def wait_for_sensors(self):
        """
        等待传感器初始化完成
        """
        while self.scan is None or self.odom is None:
            rclpy.spin_once(self.node, timeout_sec=0.1)
            time.sleep(0.01)

    def reset_robot(self):
        """
        重置机器人到原点，并生成随机目标点
        """
        self.step_count = 0

        # 随机生成目标
        GOAL_X_MIN, GOAL_X_MAX = -10.0, 8.0
        GOAL_Y_MIN, GOAL_Y_MAX = 4.0, 5.0
        gx = random.uniform(GOAL_X_MIN, GOAL_X_MAX)
        gy = random.uniform(GOAL_Y_MIN, GOAL_Y_MAX)
        self.goal = np.array([gx, gy], dtype=np.float32)

        goal_msg = PoseStamped()
        goal_msg.header.frame_id = 'map'
        goal_msg.pose.position.x = float(self.goal[0])
        goal_msg.pose.position.y = float(self.goal[1])
        goal_msg.pose.position.z = 0.0
        self.goal_pub.publish(goal_msg)
        time.sleep(0.1)

    def get_state(self):
        if self.scan is None or self.odom is None:
            return np.zeros(self.state_dim, dtype=np.float32)

        x = self.odom.pose.pose.position.x
        y = self.odom.pose.pose.position.y
        q = self.odom.pose.pose.orientation
        yaw = math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))

        dx = self.goal[0] - x
        dy = self.goal[1] - y
        distance = math.sqrt(dx*dx + dy*dy)
        angle_to_goal = math.atan2(dy, dx) - yaw
        angle_to_goal = (angle_to_goal + np.pi) % (2*np.pi) - np.pi

        vx = self.odom.twist.twist.linear.x
        vz = self.odom.twist.twist.angular.z

        state = np.concatenate([self.scan / 8.0, [vx, vz, distance, angle_to_goal]])
        return state.astype(np.float32)

    def step(self, action):
        # 等待传感器初始化
        self.wait_for_sensors()

        cmd = Twist()
        cmd.linear.x = float(action[0])
        cmd.angular.z = float(action[1])
        self.cmd_pub.publish(cmd)

        # 等待下一步
        rclpy.spin_once(self.node, timeout_sec=0.1)

        state = self.get_state()
        reward = 0.0
        done = False
        self.step_count += 1

        x = self.odom.pose.pose.position.x
        y = self.odom.pose.pose.position.y
        dist = math.hypot(self.goal[0]-x, self.goal[1]-y)

        if dist < 0.2:
            reward = 100.0
            done = True
        elif np.min(self.scan) < 0.15:
            reward = -100.0
            done = True
        elif self.step_count >= self.max_steps:
            done = True

        reward += -dist  # 鼓励靠近目标

        if done:
            self.reset_robot()

        return state, reward, done, {}

    def reset(self):
        self.reset_robot()
        self.wait_for_sensors()
        return self.get_state()
