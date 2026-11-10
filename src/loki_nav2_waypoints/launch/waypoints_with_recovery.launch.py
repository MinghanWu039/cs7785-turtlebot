"""
Lab 5: start the recovery server and the waypoint publisher together.

Start Nav2 first (nav2_bringup with turtlebot3_recovery_lab's nav2_params.yaml) and
set the initial pose; the waypoint publisher waits for an AMCL pose before sending
the first waypoint.

  ros2 launch loki_nav2_waypoints waypoints_with_recovery.launch.py
  ros2 launch loki_nav2_waypoints waypoints_with_recovery.launch.py \
      waypoints_file:=$HOME/cs7785-turtlebot/src/loki_nav2_waypoints/waypoints.txt
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    waypoints_file = LaunchConfiguration('waypoints_file')
    goal_pause = LaunchConfiguration('goal_pause')
    max_retries = LaunchConfiguration('max_retries')

    # Answers /divert_robot for the BeepAndDivertBridge node in the recovery BT.
    recovery_node = Node(
        package='turtlebot3_recovery_lab',
        executable='recovery.py',
        name='student_recovery_server',
        output='screen',
    )

    # Sends waypoints.txt to Nav2 on /goal_pose, one goal at a time.
    waypoint_node = Node(
        package='loki_nav2_waypoints',
        executable='waypoint_publisher',
        name='waypoint_publisher',
        output='screen',
        # Explicit types: launch would otherwise YAML-parse the strings, turning
        # an empty waypoints_file into None.
        parameters=[{
            'waypoints_file': ParameterValue(waypoints_file, value_type=str),
            'goal_pause': ParameterValue(goal_pause, value_type=float),
            'max_retries': ParameterValue(max_retries, value_type=int),
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'waypoints_file', default_value='',
            description='Waypoints file; empty uses the waypoints.txt installed with this package'),
        DeclareLaunchArgument(
            'goal_pause', default_value='2.0',
            description='Seconds to wait at a reached waypoint before sending the next'),
        DeclareLaunchArgument(
            'max_retries', default_value='2',
            description='Times to resend an aborted or canceled waypoint before skipping it'),

        recovery_node,
        waypoint_node,
    ])
