from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    detect_object_node = Node(
        package='loki_chase_object',
        executable='detect_object',
        name='detect_object',
        output='screen',
    )

    get_object_range_node = Node(
        package='loki_chase_object',
        executable='get_object_range',
        name='get_object_range',
        output='screen',
    )

    chase_object_node = Node(
        package='loki_chase_object',
        executable='chase_object',
        name='chase_object',
        output='screen',
    )

    return LaunchDescription([
        detect_object_node,
        get_object_range_node,
        chase_object_node,
    ])
