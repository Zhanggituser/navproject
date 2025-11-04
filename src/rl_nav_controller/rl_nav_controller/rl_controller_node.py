import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path
import numpy as np

class RLControllerNode(Node):
    """
    ROS2 节点：管理随机目标点、/plan订阅和cmd_vel发布
    """
    def __init__(self):
        super().__init__('rl_controller_node')
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.plan_sub = self.create_subscription(Path, '/plan', self.plan_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.goal = np.array([1.0, 1.0])
        self.publish_new_goal()

    def plan_callback(self, msg):
        pass  # 可处理全局路径信息

    def publish_new_goal(self):
        self.goal = np.random.uniform(-2.0, 2.0, size=(2,))
        goal_msg = PoseStamped()
        goal_msg.header.frame_id = 'map'
        goal_msg.pose.position.x = float(self.goal[0])
        goal_msg.pose.position.y = float(self.goal[1])
        goal_msg.pose.position.z = 0.0
        self.goal_pub.publish(goal_msg)
        self.get_logger().info(f'New goal: x={self.goal[0]:.2f}, y={self.goal[1]:.2f}')

def main(args=None):
    rclpy.init(args=args)
    node = RLControllerNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
