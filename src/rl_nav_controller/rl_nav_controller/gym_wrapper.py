import gym
from gym import spaces
import numpy as np
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
import math
import time
import random
import rclpy
from rclpy.node import Node
import cv2


class RLNavEnv(gym.Env):
    """
    强化学习导航环境（PPO训练 + Nav2全局路径辅助）

    ✅ 新增功能：
    - 订阅 Nav2 全局路径（/plan）
    - 沿全局路径行进获得奖励
    - 继续包含避障与目标奖励机制
    """

    metadata = {'render.modes': ['human']}

    def __init__(self):
        super().__init__()

        if not rclpy.ok():
            raise RuntimeError("Call rclpy.init() before creating RLNavEnv.")

        self.node = Node("rl_nav_env")

        # === 地图信息 ===
        self.map_path = "/home/yuzhang/ROS2/nav2_chapt7/nav2_chapt7_ws/src/fishbot_navigation2/maps/room2.pgm"
        self.map_resolution = 0.05
        self.map_origin_x = -4.94
        self.map_origin_y = -4.91
        self.occupied_thresh = 0.65
        self._load_map()

        # === ROS 接口 ===
        self.cmd_pub = self.node.create_publisher(Twist, '/cmd_vel', 10)
        self.goal_pub = self.node.create_publisher(PoseStamped, '/goal_pose', 10)
        self.goal_marker_pub = self.node.create_publisher(Marker, '/goal_marker', 10)
        self.node.create_subscription(LaserScan, '/scan', self._scan_cb, 10)
        self.node.create_subscription(Odometry, '/odom', self._odom_cb, 10)
        self.node.create_subscription(Path, '/plan', self._plan_cb, 10)  # ✅ 新增
        self.scan = None
        self.odom = None
        self.global_path = []  # ✅ 全局路径存储

        # === 环境参数 ===
        self.max_steps = 5000
        self.step_count = 0
        self.episode_reward = 0.0
        self.goal = np.array([0.0, 0.0], dtype=np.float32)
        self.prev_dist = None

        # === 动作与观测空间 ===
        self.action_space = spaces.Box(
            low=np.array([-0.2, -1.0]),
            high=np.array([0.3, 1.0]),
            dtype=np.float32
        )
        self.laser_dim = 180
        self.state_dim = self.laser_dim + 4  # laser + vx + vz + dist + angle
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.state_dim,),
            dtype=np.float32
        )

    # -------------------------
    # 基础回调
    # -------------------------
    def _load_map(self):
        img = cv2.imread(self.map_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Map not found: {self.map_path}")
        height, width = img.shape
        normalized = img.astype(np.float32) / 255.0
        free_mask = normalized >= (1.0 - self.occupied_thresh + 0.1)
        free_pixels = np.column_stack(np.where(free_mask))
        self.free_goals = []
        for py, px in free_pixels:
            wx = self.map_origin_x + px * self.map_resolution
            wy = self.map_origin_y + (height - 1 - py) * self.map_resolution
            self.free_goals.append([wx, wy])
        self.node.get_logger().info(f"✅ Map loaded: {len(self.free_goals)} free points")

    def _scan_cb(self, msg):
        scan = np.array(msg.ranges[:self.laser_dim])
        scan[np.isnan(scan)] = 8.0
        scan[np.isinf(scan)] = 8.0
        self.scan = scan

    def _odom_cb(self, msg):
        self.odom = msg

    def _plan_cb(self, msg):
        """订阅全局路径"""
        self.global_path = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        if len(self.global_path) > 0:
            self.node.get_logger().info(f"📍 Received global path with {len(self.global_path)} points.")
            self.reset()

    def _wait_sensors(self, timeout_sec=2.0):
        start = time.time()
        while (self.scan is None or self.odom is None) and (time.time() - start) < timeout_sec:
            rclpy.spin_once(self.node, timeout_sec=0.01)

    # -------------------------
    # 重置与目标发布
    # -------------------------
    def _publish_goal(self):
        goal_msg = PoseStamped()
        goal_msg.header.frame_id = "map"
        goal_msg.header.stamp = self.node.get_clock().now().to_msg()
        goal_msg.pose.position.x = float(self.goal[0])
        goal_msg.pose.position.y = float(self.goal[1])
        goal_msg.pose.orientation.w = 1.0
        self.goal_pub.publish(goal_msg)

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
        marker.scale.x = marker.scale.y = marker.scale.z = 0.2
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = 1.0, 0.0, 0.0, 1.0
        marker.lifetime = rclpy.duration.Duration(seconds=0).to_msg()
        self.goal_marker_pub.publish(marker)

    def reset(self, **kwargs):
        idx = random.randint(0, len(self.free_goals) - 1)
        self.goal = np.array(self.free_goals[idx], dtype=np.float32)
        for _ in range(3):
            self._publish_goal()
            time.sleep(0.05)
        self.node.get_logger().info(f"🎯 New goal: ({self.goal[0]:.2f}, {self.goal[1]:.2f})")

        self.step_count = 0
        self.episode_reward = 0.0
        self.prev_dist = None
        self._wait_sensors()
        if self.odom is not None:
            x = self.odom.pose.pose.position.x
            y = self.odom.pose.pose.position.y
            self.prev_dist = math.hypot(self.goal[0] - x, self.goal[1] - y)
        return self._get_obs(), {}

    # -------------------------
    # 状态与奖励
    # -------------------------
    def _get_obs(self):
        if self.scan is None or self.odom is None:
            return np.zeros(self.state_dim, dtype=np.float32)
        x = self.odom.pose.pose.position.x
        y = self.odom.pose.pose.position.y
        q = self.odom.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y ** 2 + q.z ** 2))
        dx, dy = self.goal[0] - x, self.goal[1] - y
        distance = math.hypot(dx, dy)
        angle_to_goal = math.atan2(dy, dx) - yaw
        angle_to_goal = (angle_to_goal + math.pi) % (2 * math.pi) - math.pi
        vx = self.odom.twist.twist.linear.x
        vz = self.odom.twist.twist.angular.z
        state = np.concatenate([self.scan / 8.0, [vx, vz, distance, angle_to_goal]])
        return np.nan_to_num(state, nan=0.0, posinf=8.0, neginf=0.0).astype(np.float32)

    def _compute_path_reward(self, x, y):
        """沿全局路径的奖励：离路径越近奖励越高"""
        if not self.global_path:
            return 0.0
        distances = [math.hypot(px - x, py - y) for px, py in self.global_path]
        min_d = min(distances)
        if min_d < 0.3:
            return (0.3 - min_d) * 5.0  # 奖励上限约 1.5
        return 0.0

    # -------------------------
    # 主 step()
    # -------------------------
    def step(self, action):
        cmd = Twist()
        cmd.linear.x = float(np.clip(action[0], -0.2, 0.3))
        cmd.angular.z = float(np.clip(action[1], -1.0, 1.0))
        self.cmd_pub.publish(cmd)
        rclpy.spin_once(self.node, timeout_sec=0.01)
        time.sleep(0.01)
        self.step_count += 1

        state = self._get_obs()
        reward = 0.0
        done = False

        if self.odom is None:
            reward = -50.0
            done = True
        else:
            x = self.odom.pose.pose.position.x
            y = self.odom.pose.pose.position.y
            dist_to_goal = math.hypot(self.goal[0] - x, self.goal[1] - y)
            min_scan = np.min(self.scan) if self.scan is not None else 8.0

            # === 终止条件 ===
            if dist_to_goal < 0.2:
                reward = 200.0
                done = True
                self.node.get_logger().info("✅ Reached goal")
            elif self.step_count >= self.max_steps:
                reward = -20.0
                done = True
                self.node.get_logger().info("⏰ Timeout")
            elif min_scan < 0.2:
                reward = -100.0
                done = False
                self.node.get_logger().warn(f"⚠️ Collision! min_laser={min_scan:.2f}")

            # === 进行中 ===
            else:
                if self.prev_dist is not None:
                    reward += (self.prev_dist - dist_to_goal) * 10.0
                if min_scan < 0.5:
                    reward -= (0.5 - min_scan) * 20.0
                if min_scan < 0.35 and action[0] < 0:
                    reward += 5.0

                # ✅ 路径奖励
                reward += self._compute_path_reward(x, y)
                self.prev_dist = dist_to_goal

        self.episode_reward += reward

        if self.step_count % 100 == 0:
            self.node.get_logger().info(f"🕒 Step {self.step_count}, total reward={self.episode_reward:.2f}")

        return state, reward, done, {}

    def close(self):
        if hasattr(self, 'node') and self.node:
            self.node.destroy_node()
