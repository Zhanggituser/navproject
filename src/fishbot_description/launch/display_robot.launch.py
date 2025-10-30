import launch
import launch_ros
from ament_index_python.packages import get_package_share_directory
import os
import launch_ros.parameter_descriptions


def generate_launch_description():
    """
    该函数由 ROS 2 launch 系统调用，用于生成整个 LaunchDescription（即启动描述）。
    它定义了在运行该 launch 文件时需要启动的节点、参数和配置。
    """

    # ===============================
    # 1️⃣ 获取默认的 URDF 文件路径
    # ===============================
    # 通过 ament_index_python 获取指定功能包（fishbot_description）的共享目录路径
    # 假设包结构类似：
    # fishbot_description/
    # ├── urdf/
    # │   └── fish_robot.urdf
    # └── package.xml
    #
    # 最终路径形如：
    # /home/ros2_ws/install/fishbot_description/share/fishbot_description/urdf/fist_robot.urdf
    urdf_package_path = get_package_share_directory("fishbot_description")
    default_urdf_path = os.path.join(urdf_package_path, 'urdf', 'first_robot.urdf')
    default_rviz_path = os.path.join(urdf_package_path, 'config', 'display_robot_model.rviz')

    # =========================================
    # 2️⃣ 声明一个可修改的“模型路径”参数（Launch Argument）
    # =========================================
    # 允许用户在命令行通过如下方式覆盖默认路径：
    #   ros2 launch fishbot_description display.launch.py model:=<你的新URDF路径>
    #
    # 这样该 launch 文件可以复用于不同的机器人模型。
    action_declare_arg_mode_path = launch.actions.DeclareLaunchArgument(
        name='model',                             # 参数名为 'model'
        default_value=str(default_urdf_path),     # 默认值为上面获取的 fish_robot.urdf
        description='加载我的模型文件路径'          # 参数描述（在 --show-args 时显示）
    )


    # =========================================
    # 3️⃣ 使用 Command 替代 cat 命令读取 URDF 内容
    # =========================================
    # Command substitution 会在启动时执行 shell 命令，这里执行的是：
    #   cat <model路径>
    # 用于把 URDF 文件的文本内容读出来，供 robot_state_publisher 使用。
    #
    # LaunchConfiguration('model') 代表上一步中声明的参数值。

    # 1
    # substitutions_command_result = launch.substitutions.Command(
    #     [
    #         'cat ',  # 调用系统命令 cat
    #         launch.substitutions.LaunchConfiguration('model')
    #     ]
    # )

    substitutions_command_result = launch.substitutions.Command(
        [
            'xacro ',  # 调用系统命令 cat
            launch.substitutions.LaunchConfiguration('model')
        ]
    )








    # =========================================
    # 4️⃣ 将 URDF 内容转换为 robot_description 参数值
    # =========================================
    # robot_state_publisher 节点需要一个名为 "robot_description" 的参数，
    # 其中包含完整的 URDF 文本（而不是文件路径）。
    #
    # ParameterValue 可以接收动态替换的内容（如 Command 的输出）。
    robot_description_value = launch_ros.parameter_descriptions.ParameterValue(
        substitutions_command_result,
        value_type=str
    )


    # =========================================
    # 5️⃣ 启动 robot_state_publisher 节点
    # =========================================
    # 该节点根据 robot_description（URDF 模型）计算各连杆之间的 TF 变换。
    # 它是 RViz 可显示机器人模型的关键节点。
    #
    # robot_state_publisher 会订阅 /joint_states 话题，
    # 并发布 TF 树（即各部件之间的位置与姿态关系）。
    action_robot_state_publisher = launch_ros.actions.Node(
        package='robot_state_publisher',       # 节点所在包
        executable='robot_state_publisher',    # 可执行文件
        parameters=[{'robot_description': robot_description_value}]  # 传入URDF文本参数
    )


    # =========================================
    # 6️⃣ 启动 joint_state_publisher 节点
    # =========================================
    # 该节点负责发布 /joint_states 话题，
    # 它会根据 URDF 文件中的关节信息生成滑动条（在 GUI 模式下），
    # 方便手动调节每个关节的角度，用于调试。
    #
    # 若模型包含关节，则需要该节点；若模型是静态的，可以省略。
    action_joint_state_publisher = launch_ros.actions.Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        # parameters=[{'use_sim_time': True}] # 可选参数，若在 Gazebo 等仿真中运行
    )


    # =========================================
    # 7️⃣ 启动 RViz2 可视化工具
    # =========================================
    # 用于在 3D 界面中显示 TF 树、机器人模型、传感器数据等。
    #
    # 你也可以加载自定义配置文件，例如 default.rviz：
    # arguments=['-d', '<路径>/default.rviz']
    action_rviz_node = launch_ros.actions.Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', default_rviz_path],
    )


    # =========================================
    # 8️⃣ 返回 LaunchDescription（总启动配置）
    # =========================================
    # 把所有声明和节点加入 LaunchDescription 中，
    # launch 系统会按顺序执行这些动作。
    return launch.LaunchDescription([
        action_declare_arg_mode_path,       # 声明命令行参数
        action_robot_state_publisher,       # 启动 TF 发布节点
        action_joint_state_publisher,       # 启动关节状态发布节点
        action_rviz_node,                   # 启动 RViz 可视化界面
    ])
