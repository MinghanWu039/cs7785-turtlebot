import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    display = LaunchConfiguration('display', default=False)
    image_topic = LaunchConfiguration('image_topic', default='/image_raw/compressed')
    static = LaunchConfiguration('static', default=False)

    hardcoded_hue = LaunchConfiguration('hardcoded_hue', default=False)
    color_profile_path = LaunchConfiguration(
        'color_profile_path', default=os.path.expanduser('~/.loki_color_profile.json'))
    hue_lower = LaunchConfiguration('hue_lower', default=-1.0)
    hue_upper = LaunchConfiguration('hue_upper', default=-1.0)
    min_saturation = LaunchConfiguration('min_saturation', default=60)
    min_value = LaunchConfiguration('min_value', default=40)

    find_object_node = Node(
        package='loki_object_follower',
        executable='find_object',
        name='find_object',
        output='screen',
        parameters=[{
            'display': display,
            'image_topic': image_topic,
            'static': static,
            'hardcoded_hue': hardcoded_hue,
            'color_profile_path': color_profile_path,
            'hue_lower': hue_lower,
            'hue_upper': hue_upper,
            'min_saturation': min_saturation,
            'min_value': min_value,
        }],
    )

    rotate_robot_node = Node(
        package='loki_object_follower',
        executable='rotate_robot',
        name='rotate_robot',
        output='screen',
    )

    return LaunchDescription([
        DeclareLaunchArgument('display', default_value='false'),
        DeclareLaunchArgument('image_topic', default_value='/image_raw/compressed'),
        DeclareLaunchArgument('static', default_value='false'),
        DeclareLaunchArgument('hardcoded_hue', default_value='false'),
        DeclareLaunchArgument(
            'color_profile_path',
            default_value=os.path.expanduser('~/.loki_color_profile.json')),
        DeclareLaunchArgument('hue_lower', default_value='-1.0'),
        DeclareLaunchArgument('hue_upper', default_value='-1.0'),
        DeclareLaunchArgument('min_saturation', default_value='60'),
        DeclareLaunchArgument('min_value', default_value='40'),

        find_object_node,
        rotate_robot_node,
    ])
