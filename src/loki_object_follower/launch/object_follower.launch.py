from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
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

    return LaunchDescription([
        find_object_node,
        rotate_robot_node,
    ])
