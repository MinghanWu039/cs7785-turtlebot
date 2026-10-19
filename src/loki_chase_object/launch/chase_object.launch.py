import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    display = LaunchConfiguration('display', default=False)
    image_topic = LaunchConfiguration('image_topic', default='/image_raw/compressed')
    static = LaunchConfiguration('static', default=False)
    horizontal_fov = LaunchConfiguration('horizontal_fov', default=1.0862)

    hardcoded_hue = LaunchConfiguration('hardcoded_hue', default=False)
    color_profile_path = LaunchConfiguration(
        'color_profile_path', default=os.path.expanduser('~/.loki_color_profile.json'))
    hue_lower = LaunchConfiguration('hue_lower', default=-1.0)
    hue_upper = LaunchConfiguration('hue_upper', default=-1.0)
    min_saturation = LaunchConfiguration('min_saturation', default=60)
    min_value = LaunchConfiguration('min_value', default=40)

    desired_distance = LaunchConfiguration('desired_distance', default=0.2)

    detect_object_node = Node(
        package='loki_chase_object',
        executable='detect_object',
        name='detect_object',
        output='screen',
        parameters=[{
            'display': display,
            'image_topic': image_topic,
            'static': static,
            'horizontal_fov': horizontal_fov,
            'hardcoded_hue': hardcoded_hue,
            'color_profile_path': color_profile_path,
            'hue_lower': hue_lower,
            'hue_upper': hue_upper,
            'min_saturation': min_saturation,
            'min_value': min_value,
        }],
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
        parameters=[{
            'desired_distance': desired_distance,
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument('display', default_value='false'),
        DeclareLaunchArgument('image_topic', default_value='/image_raw/compressed'),
        DeclareLaunchArgument('static', default_value='false'),
        DeclareLaunchArgument('horizontal_fov', default_value='1.0862'),
        DeclareLaunchArgument('hardcoded_hue', default_value='false'),
        DeclareLaunchArgument(
            'color_profile_path',
            default_value=os.path.expanduser('~/.loki_color_profile.json')),
        DeclareLaunchArgument('hue_lower', default_value='-1.0'),
        DeclareLaunchArgument('hue_upper', default_value='-1.0'),
        DeclareLaunchArgument('min_saturation', default_value='60'),
        DeclareLaunchArgument('min_value', default_value='40'),
        DeclareLaunchArgument('desired_distance', default_value='0.2'),

        detect_object_node,
        get_object_range_node,
        chase_object_node,
    ])
