#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import torch
import numpy as np
import math
import torch.nn as nn


class RLControllerNode(Node):
    """
    ROS2 节点：加载 PPO 模型，仅响应外部 /goal_pose（如 RViz 的 2D Nav Goal）
    不再自动发布目标 → RViz 中目标点不再漂移！
    """
    def __init__(self):
        super().__init__('rl_controller_node')

        self.declare_parameter('model_path', 'final_model.pth')
        self.model_path = self.get_parameter('model_path').value

        self.load_rl_model()

        # 关键：只订阅 /goal_pose，不发布！
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.goal_sub = self.create_subscription(
            PoseStamped, '/goal_pose', self.goal_callback, 10
        )
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        self.scan = None
        self.odom = None
        self.goal = None
        self.has_goal = False

        self.timer = self.create_timer(0.1, self.control_loop)  # 10 Hz
        self.get_logger().info("RL Controller started. Waiting for /goal_pose...")

    def load_rl_model(self):
        # 状态维度必须与训练一致
        state_dim = 184  # 180 (laser) + 4 (vx, vz, dist, angle)
        action_dim = 2

        self.actor_net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.actor_mean = nn.Linear(128, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))

        # 加载模型
        checkpoint = torch.load(self.model_path, map_location='cpu')
        self.actor_net.load_state_dict({
            k.replace('fc.', ''): v for k, v in checkpoint.items() if 'fc' in k
        }, strict=False)
        self.actor_mean.load_state_dict({
            'weight': checkpoint['actor_mean.weight'],
            'bias': checkpoint['actor_mean.bias']
        })
        if 'actor_log_std' in checkpoint:
            self.actor_log_std.data = checkpoint['actor_log_std']

        self.actor_net.eval()
        self.actor_mean.eval()
        self.get_logger().info(f"Model loaded: {self.model_path}")

    def scan_callback(self, msg):
        scan = np.array(msg.ranges[:180])
        scan[np.isinf(scan)] = 8.0
        self.scan = scan / 8.0

    def odom_callback(self, msg):
        self.odom = msg

    def goal_callback(self, msg):
        """接收 RViz 的 2D Nav Goal 或其他外部目标"""
        self.goal = np.array([
            msg.pose.position.x,
            msg.pose.position.y
        ], dtype=np.float32)
        self.has_goal = True
        self.get_logger().info(f"New goal received: ({self.goal[0]:.2f}, {self.goal[1]:.2f})")

    def get_state(self):
        if self.scan is None or self.odom is None or self.goal is None:
            return None

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

        return np.concatenate([self.scan, [vx, vz, distance, angle_to_goal]]).astype(np.float32)

    def control_loop(self):
        # 无目标或无传感器时停止
        if not self.has_goal or self.scan is None or self.odom is None:
            cmd = Twist()
            self.cmd_pub.publish(cmd)
            return

        state = self.get_state()
        if state is None:
            return

        with torch.no_grad():
            x = torch.FloatTensor(state).unsqueeze(0)
            features = self.actor_net(x)
            action_mean = self.actor_mean(features)
            action = action_mean.squeeze(0).numpy()

        # 关键：降低速度提高安全性！
        cmd = Twist()
        cmd.linear.x = float(np.clip(action[0], 0.0, 0.2))   # 从 0.3 → 0.2
        cmd.angular.z = float(np.clip(action[1], -0.8, 0.8)) # 从 ±1.0 → ±0.8
        self.cmd_pub.publish(cmd)

        # 到达目标后保留目标（不发布新目标，不漂移）
        if self.odom is not None:
            x = self.odom.pose.pose.position.x
            y = self.odom.pose.pose.position.y
            dist = math.hypot(self.goal[0] - x, self.goal[1] - y)
            if dist < 0.3:
                self.get_logger().info("Goal reached!")

    def destroy_node(self):
        stop_cmd = Twist()
        self.cmd_pub.publish(stop_cmd)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RLControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()