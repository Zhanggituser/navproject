import gym
from gym import spaces
import numpy as np
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
import math
import time
import random
import rclpy
from rclpy.node import Node


class RLNavEnv(gym.Env):
    """
    自定义强化学习环境（PPO训练局部路径控制器）
    注意：ROS 2 初始化（rclpy.init()）应在外部调用！
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(RLNavEnv, self).__init__()

        if not rclpy.ok():
            raise RuntimeError("ROS 2 not initialized. Call rclpy.init() before creating RLNavEnv.")

        self.node = Node("rl_env_node")

        self.cmd_pub = self.node.create_publisher(Twist, '/cmd_vel', 10)
        self.goal_pub = self.node.create_publisher(PoseStamped, '/goal_pose', 10)
        self.goal_marker_pub = self.node.create_publisher(Marker, '/goal_marker', 10)
        self.robot_pose_pub = self.node.create_publisher(Marker, '/robot_marker', 10)

        self.scan = None
        self.odom = None
        self.scan_msg_stamp = None
        self.odom_msg_stamp = None
        self.node.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.node.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        self.max_steps = 500
        self.step_count = 0
        self.goal = np.array([1.0, 1.0], dtype=np.float32)
        self.prev_dist = float('inf')  # 用于奖励计算

        self.laser_dim = 180
        self.state_dim = self.laser_dim + 4
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0]),
            high=np.array([0.3, 1.0]),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )

    def scan_callback(self, msg):
        self.scan = np.array(msg.ranges[:self.laser_dim])
        self.scan[np.isinf(self.scan)] = 8.0
        self.scan_msg_stamp = msg.header.stamp

    def odom_callback(self, msg):
        self.odom = msg
        self.odom_msg_stamp = msg.header.stamp

    def wait_for_sensors(self, timeout_sec=1.0):
        start = time.time()
        while (self.scan is None or self.odom is None) and (time.time() - start) < timeout_sec:
            rclpy.spin_once(self.node, timeout_sec=0.01)
        if self.scan is None or self.odom is None:
            self.node.get_logger().warn("Timeout waiting for sensor data!")

    def publish_goal_marker(self):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.node.get_clock().now().to_msg()
        marker.ns = "goal"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = float(self.goal[0])
        marker.pose.position.y = float(self.goal[1])
        marker.pose.position.z = 0.1
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        self.goal_marker_pub.publish(marker)

    def publish_robot_marker(self):
        if self.odom is None:
            return
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.node.get_clock().now().to_msg()
        marker.ns = "robot"
        marker.id = 1
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose = self.odom.pose.pose
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.2
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        self.robot_pose_pub.publish(marker)

    def reset_robot(self):
        self.step_count = 0
        self.prev_dist = float('inf')

        # 更合理的训练区域（窄走廊）
        GOAL_X_MIN, GOAL_X_MAX = -10.0, 8.0
        GOAL_Y_MIN, GOAL_Y_MAX = -4.0, 4.0
        gx = random.uniform(GOAL_X_MIN, GOAL_X_MAX)
        gy = random.uniform(GOAL_Y_MIN, GOAL_Y_MAX)
        self.goal = np.array([gx, gy], dtype=np.float32)

        goal_msg = PoseStamped()
        goal_msg.header.frame_id = 'map'
        goal_msg.pose.position.x = float(self.goal[0])
        goal_msg.pose.position.y = float(self.goal[1])
        goal_msg.pose.position.z = 0.0
        self.node.get_logger().info(f'Goal position set to: ({self.goal[0]:.2f}, {self.goal[1]:.2f})')  
        self.goal_pub.publish(goal_msg)
        self.publish_goal_marker()

    def get_state(self):
        if self.scan is None or self.odom is None:
            return np.zeros(self.state_dim, dtype=np.float32)

        x = self.odom.pose.pose.position.x
        y = self.odom.pose.pose.position.y
        q = self.odom.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y),
                         1 - 2 * (q.y * q.y + q.z * q.z))

        dx = self.goal[0] - x
        dy = self.goal[1] - y
        distance = math.hypot(dx, dy)
        angle_to_goal = math.atan2(dy, dx) - yaw
        angle_to_goal = (angle_to_goal + math.pi) % (2 * math.pi) - math.pi

        vx = self.odom.twist.twist.linear.x
        vz = self.odom.twist.twist.angular.z

        normalized_scan = self.scan / 8.0
        state = np.concatenate([normalized_scan, [vx, vz, distance, angle_to_goal]])
        return state.astype(np.float32)

    def step(self, action):
        cmd = Twist()
        cmd.linear.x = float(np.clip(action[0], 0.0, 0.3))
        cmd.angular.z = float(np.clip(action[1], -1.0, 1.0))
        self.cmd_pub.publish(cmd)

        rclpy.spin_once(self.node, timeout_sec=0.01)
        time.sleep(0.01)

        self.step_count += 1
        state = self.get_state()
        reward = 0.0
        done = False
        truncated = False

        if self.odom is None:
            reward = -20.0
            done = True
        else:
            x = self.odom.pose.pose.position.x
            y = self.odom.pose.pose.position.y
            dist_to_goal = math.hypot(self.goal[0] - x, self.goal[1] - y)

            if dist_to_goal < 0.2:
                reward = 50.0
                done = True
            elif self.scan is not None and np.min(self.scan) < 0.3:  # 提前避障
                reward = -20.0
                done = True
            elif self.step_count >= self.max_steps:
                reward = -10.0
                done = True
                truncated = True
            else:
                # 核心奖励改进
                reward += (self.prev_dist - dist_to_goal) * 5.0  # 前进奖励
                min_scan = np.min(self.scan) if self.scan is not None else 8.0
                reward += min_scan * 0.5  # 避障奖励
                # 惩罚无效动作
                if abs(action[0]) < 0.05 and abs(action[1]) > 0.8:
                    reward -= 0.1

            self.prev_dist = dist_to_goal

        self.publish_robot_marker()
        return state, reward, done, {"TimeLimit.truncated": truncated}

    def reset(self, **kwargs):
        """兼容新旧 Gym API"""
        self.reset_robot()
        self.wait_for_sensors()
        return self.get_state(), {}

    def close(self):
        if hasattr(self, 'node') and self.node:
            self.node.destroy_node()