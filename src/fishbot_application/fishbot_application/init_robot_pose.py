from geometry_msgs.msg import PoseStamped
from nav2_simple_commander.robot_navigator import BasicNavigator
import rclpy



def main():
    rclpy.init()
    node = rclpy.create_node('init_robot_pose')
    navigator = BasicNavigator()
    # 等待 Nav2 启动
    navigator.waitUntilNav2Active()
    # 创建初始位姿消息
    initial_pose = PoseStamped()
    initial_pose.header.frame_id = 'map'
    initial_pose.header.stamp = node.get_clock().now().to_msg()
    initial_pose.pose.position.x = 0.0
    initial_pose.pose.position.y = 0.0
    initial_pose.pose.position.z = 0.0
    initial_pose.pose.orientation.w = 0.0

    # 设置机器人的初始位姿
    navigator.setInitialPose(initial_pose)

    node.get_logger().info('Initial robot pose has been set.')

    rclpy.shutdown()