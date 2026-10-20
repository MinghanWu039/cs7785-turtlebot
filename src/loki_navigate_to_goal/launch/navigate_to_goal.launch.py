from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    viz = LaunchConfiguration('viz', default='true')

    vlio_node = Node(
        package='loki_navigate_to_goal',
        executable='vlio_node',
        name='vlio_node',
        output='screen',
    )

    go_to_goal_node = Node(
        package='loki_navigate_to_goal',
        executable='go_to_goal',
        name='go_to_goal',
        output='screen',
    )

    # Streams /vlio_odom and the scan map to scripts/odom_viz_client.py over TCP.
    odom_viz_server_node = Node(
        package='loki_navigate_to_goal',
        executable='odom_viz_server',
        name='odom_viz_server',
        output='screen',
        parameters=[{'odom_topic': '/vlio_odom'}],
        condition=IfCondition(viz),
    )

    return LaunchDescription([
        DeclareLaunchArgument('viz', default_value='true'),

        vlio_node,
        go_to_goal_node,
        odom_viz_server_node,
    ])
