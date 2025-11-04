import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory  # <--- 这一行必须加

def generate_launch_description():

    # ---------------- 参数 ----------------
    map_yaml_file = LaunchConfiguration('map')
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    params_file = LaunchConfiguration('params_file')

    # ---------------- Nav2 launch ----------------
    nav2_launch_dir = os.path.join(
        get_package_share_directory('nav2_bringup'),
        'launch'
    )
    nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav2_launch_dir, 'navigation_launch.py')
        ),
        launch_arguments={
            'map': map_yaml_file,
            'use_sim_time': use_sim_time,
            'params_file': params_file
        }.items()
    )

    # ---------------- RL 控制器节点 ----------------
    rl_controller_node = Node(
        package='rl_nav_controller',
        executable='rl_controller_node',
        name='rl_controller_node',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'scan_topic': '/scan',
            'imu_topic': '/imu',
            'odom_topic': '/odom',
            'path_topic': '/plan',  # Nav2 全局路径
            'cmd_topic': '/cmd_vel',
            'control_rate': 10.0,
            'scan_vector_length': 64,
            'goal_horizon': 5
        }]
    )

    return LaunchDescription([
        DeclareLaunchArgument('map', default_value='/home/yuzhang/ROS2/mynav/mynav_ws/src/fishbot_description/maps/room.yaml',
                              description='Full path to map yaml file'),
        DeclareLaunchArgument('params_file',  default_value='/home/yuzhang/ROS2/mynav/mynav_ws/src/fishbot_description/config/nav2_params.yaml',
                              description='Full path to Nav2 parameters file'),
        nav2_launch,
        rl_controller_node
    ])
