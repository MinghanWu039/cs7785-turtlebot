from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # /vlio_odom plots the ICP-refined pose; pass odom_topic:=/odom to plot raw odometry.
    odom_topic = LaunchConfiguration('odom_topic', default='/vlio_odom')

    vlio_node = Node(
        package='loki_navigate_to_goal',
        executable='vlio_node',
        name='vlio_node',
        output='screen',
    )

    odom_viz_server_node = Node(
        package='loki_navigate_to_goal',
        executable='odom_viz_server',
        name='odom_viz_server',
        output='screen',
        parameters=[{'odom_topic': odom_topic}],
    )

    return LaunchDescription([
        DeclareLaunchArgument('odom_topic', default_value='/vlio_odom'),

        vlio_node,
        odom_viz_server_node,
    ])
