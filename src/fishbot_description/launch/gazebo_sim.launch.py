import launch
import launch_ros
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # 获取 share 路径
    urdf_package_path = get_package_share_directory('fishbot_description')

    default_xacro_path = os.path.join(urdf_package_path, 'urdf', 'fishbot', 'fishbot.urdf.xacro')
    # default_rviz_config_path = os.path.join(urdf_package_path, 'config', 'rviz', 'display_model.rviz')
    default_gazebo_world_path = os.path.join(urdf_package_path, 'world', 'custom_room.world')

    # 声明参数
    action_declare_arg_model_path = launch.actions.DeclareLaunchArgument(
        name='model',
        default_value=default_xacro_path,
        description='URDF 的绝对路径'
    )

    # 使用 xacro 生成 robot_description 参数
    robot_description = launch_ros.parameter_descriptions.ParameterValue(
        launch.substitutions.Command(
            ['xacro ', launch.substitutions.LaunchConfiguration('model')]
        ),
        value_type=str
    )

    # 状态发布节点
    robot_state_publisher_node = launch_ros.actions.Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_description}]
    )

    # # joint_state_publisher
    # action_joint_state_publisher = launch_ros.actions.Node(
    #     package='joint_state_publisher',
    #     executable='joint_state_publisher',
    # )

    # 启动 Gazebo
    action_launch_gazebo = launch.actions.IncludeLaunchDescription(
        launch.launch_description_sources.PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('gazebo_ros'), 'launch', 'gazebo.launch.py')
        ),
        # ❗注意这里加上 .items()，否则会报 “too many values to unpack” 错误
        launch_arguments={
            'world': default_gazebo_world_path,
            'verbose': 'true'
        }.items()
    )



    action_spawn_entity = launch_ros.actions.Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', 'robot_description',
            '-entity', 'fishbot'
        ],

    )

    # # RViz 节点（暂时注释）
    # rviz_node = launch_ros.actions.Node(
    #     package='rviz2',
    #     executable='rviz2',
    #     arguments=['-d', default_rviz_config_path]
    # )



    action_load_joint_state_controller =launch.actions.ExecuteProcess(
        cmd = 'ros2 control load_controller fishbot_joint_state_broadcaster --set-state active'.split(),
        output='screen'
    )

    action_load_effort_controller =launch.actions.ExecuteProcess(
        cmd = 'ros2 control load_controller fishbot_effort_controller --set-state active'.split(),
        output='screen'
    )
    action_load_diff_drive_controller =launch.actions.ExecuteProcess(
        cmd = 'ros2 control load_controller fishbot_diff_drive_controller --set-state active'.split(),
        output='screen'
    )

    return launch.LaunchDescription([
        action_declare_arg_model_path,
        # action_joint_state_publisher,
        robot_state_publisher_node,
        action_launch_gazebo,
        action_spawn_entity,

        # 事件动作，当加载机器人结束后执行    
        launch.actions.RegisterEventHandler(
            event_handler=launch.event_handlers.OnProcessExit(
                target_action=action_spawn_entity,
                on_exit=[action_load_joint_state_controller],)
            ),
        # 事件动作，load_fishbot_diff_drive_controller
        launch.actions.RegisterEventHandler(
        event_handler=launch.event_handlers.OnProcessExit(
            target_action=action_load_joint_state_controller,
            # on_exit=[action_load_effort_controller],)
            on_exit=[action_load_diff_drive_controller],)
            ),
       
    ])
