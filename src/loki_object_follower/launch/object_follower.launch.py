import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    bringup_dir = get_package_share_directory('turtlebot3_bringup')
    use_lidar = LaunchConfiguration('use_lidar')
    usb_port = LaunchConfiguration('usb_port')

    use_lidar_arg = DeclareLaunchArgument(
        'use_lidar',
        default_value='false',
        description='Also start the LiDAR driver (needs LDS_MODEL set)')
    usb_port_arg = DeclareLaunchArgument(
        'usb_port',
        default_value='/dev/ttyACM0',
        description='USB port connected to the OpenCR board')

    # With the LiDAR: the stock bringup (requires TURTLEBOT3_MODEL and LDS_MODEL).
    bringup_with_lidar = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(bringup_dir, 'launch', 'robot.launch.py')),
        launch_arguments={'usb_port': usb_port}.items(),
        condition=IfCondition(use_lidar),
    )

    # Without the LiDAR: same as robot.launch.py minus the LiDAR driver.
    bringup_no_lidar = [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(
                    bringup_dir, 'launch', 'turtlebot3_state_publisher.launch.py')),
            launch_arguments={'namespace': ''}.items(),
            condition=UnlessCondition(use_lidar),
        ),
        Node(
            package='turtlebot3_node',
            executable='turtlebot3_ros',
            parameters=[os.path.join(
                bringup_dir, 'param', os.environ['TURTLEBOT3_MODEL'] + '.yaml')],
            arguments=['-i', usb_port],
            output='screen',
            condition=UnlessCondition(use_lidar),
        ),
    ]

    find_object_node = Node(
        package='loki_object_follower',
        executable='find_object',
        name='find_object',
        output='screen',
    )

    rotate_robot_node = Node(
        package='loki_object_follower',
        executable='rotate_robot',
        name='rotate_robot',
        output='screen',
    )

    # PLACEHOLDER: camera node.
    # find_object subscribes to /camera/image_raw/compressed (sensor_msgs/CompressedImage).
    # Swap in the real camera driver once it's decided, e.g.:
    # camera_node = Node(
    #     package='<camera_package>',
    #     executable='<camera_executable>',
    #     name='camera',
    #     output='screen',
    # )

    return LaunchDescription([
        use_lidar_arg,
        usb_port_arg,
        bringup_with_lidar,
        *bringup_no_lidar,
        find_object_node,
        rotate_robot_node,
        # camera_node,
    ])
