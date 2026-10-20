from collections import deque
import math
import os
import re

from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import LaserScan

from loki_navigate_to_goal.geometry import (
    scan_to_points, stamp_seconds, transform, wrap_angle, yaw_from_quaternion)
from loki_navigate_to_goal.hybrid_control import compute_control_data, State, StateMachine


POSE_TIMEOUT = 1.0  # s, stop if no /vlio_odom has arrived for this long
SCAN_BUFFER = 10  # recent scans kept for matching a /vlio_odom stamp
STATE_LOG_PERIOD = 1.0  # s between current-state log lines; transitions are always logged

ANGULAR_KP = 1.5  # rad/s per rad of heading error
ANGULAR_MAX_SPEED = 1.5  # rad/s

LINEAR_KP = 0.5  # m/s per m of distance to the goal
LINEAR_MAX_SPEED = 0.15  # m/s
BOUNDARY_SPEED = 0.08  # m/s while avoiding or following an obstacle


def load_waypoints(path):
    """Read 'x y' or 'x,y' per line; blank lines and lines starting with '#' are skipped."""
    waypoints = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            x, y = (float(v) for v in re.split(r'[,\s]+', line)[:2])
            waypoints.append(np.array([x, y]))
    return waypoints


def clamp(value, limit):
    return max(-limit, min(limit, value))


class GoToGoal(Node):
    """
    Drive through the waypoints in order, avoiding obstacles seen by the lidar.

    Subscribes to /vlio_odom (pose) and /scan, and publishes Twist commands on /cmd_vel.
    On every pose, hybrid_control's StateMachine picks a state from the control vectors
    of compute_control_data; a P controller then turns towards that state's unit vector
    and drives forward, slowing to a stop while the heading error is large. At each
    waypoint the robot stops for goal_pause seconds. If no /vlio_odom arrives for
    POSE_TIMEOUT, the robot is stopped.
    """

    def __init__(self):
        super().__init__('go_to_goal')
        # Empty means the wayPoints.txt installed with this package.
        waypoints_file = self.declare_parameter('waypoints_file', '').value or os.path.join(
            get_package_share_directory('loki_navigate_to_goal'), 'wayPoints.txt')
        self.goal_tolerance = self.declare_parameter('goal_tolerance', 0.05).value  # m
        self.goal_pause = self.declare_parameter('goal_pause', 10.0).value  # s
        # base_scan sits 3.2 cm behind base_link on the burger (turtlebot3_burger.urdf).
        self.laser_x_offset = self.declare_parameter('laser_x_offset', -0.032).value

        self.waypoints = load_waypoints(waypoints_file)
        self.goal_index = 0
        self.state_machine = StateMachine(State.GO_TO_GOAL)
        self.origin = None  # first /vlio_odom pose; the waypoints are relative to it
        self.scans = deque(maxlen=SCAN_BUFFER)  # (stamp, LaserScan)
        self.pause_until = None  # time at which the stop at the current goal ends
        self.last_pose_time = None

        self.create_subscription(
            Odometry, '/vlio_odom', self.pose_callback, qos_profile_sensor_data)
        self.create_subscription(
            LaserScan, '/scan', self.scan_callback, qos_profile_sensor_data)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.timeout_timer = self.create_timer(0.1, self.timeout_callback)
        listed = ', '.join(f'({x:g}, {y:g})' for x, y in self.waypoints)
        self.get_logger().info(f'{len(self.waypoints)} waypoints from {waypoints_file}: {listed}')

    def scan_callback(self, msg):
        self.scans.append((stamp_seconds(msg.header.stamp), msg))

    def pose_callback(self, msg):
        self.last_pose_time = self.get_clock().now()
        if self.goal_index >= len(self.waypoints):
            self.publish_cmd(0.0, 0.0)
            return
        if not self.scans:
            return

        pose = self.zeroed_pose(msg)
        pos = np.array(pose[:2])
        goal = self.waypoints[self.goal_index]

        if self.pause_until is not None:
            self.publish_cmd(0.0, 0.0)
            if self.get_clock().now() < self.pause_until:
                return
            self.pause_until = None
            self.next_goal()
            if self.goal_index >= len(self.waypoints):
                return
            goal = self.waypoints[self.goal_index]

        goal_dist = np.linalg.norm(goal - pos)
        if goal_dist < self.goal_tolerance:
            self.get_logger().info(
                f'Reached waypoint {self.goal_index + 1} at ({pos[0]:.3f}, {pos[1]:.3f}), '
                f'{goal_dist * 100:.1f} cm off; stopping for {self.goal_pause:g} s')
            self.pause_until = self.get_clock().now() + Duration(seconds=self.goal_pause)
            self.publish_cmd(0.0, 0.0)
            return

        scan = self.scan_for(stamp_seconds(msg.header.stamp))
        points = transform(pose, scan_to_points(
            scan.ranges, scan.angle_min, scan.angle_increment,
            scan.range_min, scan.range_max, self.laser_x_offset))
        u_state, obs_dist, goal_dist = compute_control_data(pos, points, goal)

        previous = self.state_machine.cur_state
        state = self.state_machine.update_state(u_state, obs_dist, goal_dist)
        if state != previous:
            self.get_logger().info(
                f'{previous.name} -> {state.name} (obstacle {obs_dist:.2f} m, '
                f'goal {goal_dist:.2f} m)')
        self.get_logger().info(
            f'State {state.name}: waypoint {self.goal_index + 1} at {goal_dist:.2f} m, '
            f'obstacle {obs_dist:.2f} m, pose ({pose[0]:.2f}, {pose[1]:.2f}, '
            f'{math.degrees(pose[2]):.0f} deg)', throttle_duration_sec=STATE_LOG_PERIOD)

        u = u_state[state]
        if not u.any():
            self.publish_cmd(0.0, 0.0)  # degenerate direction; wait for the next pose
            return
        heading_error = wrap_angle(math.atan2(u[1], u[0]) - pose[2])
        angular_z = clamp(ANGULAR_KP * heading_error, ANGULAR_MAX_SPEED)
        if state == State.GO_TO_GOAL:
            linear_x = min(LINEAR_KP * goal_dist, LINEAR_MAX_SPEED)
        else:
            linear_x = BOUNDARY_SPEED
        # Turn in place while facing away from u; drive forward as the heading lines up.
        linear_x *= max(0.0, math.cos(heading_error))
        self.publish_cmd(linear_x, angular_z)

    def zeroed_pose(self, msg):
        """The /vlio_odom pose relative to the first one, so the run starts at (0, 0) facing +x."""
        p = msg.pose.pose
        x, y, yaw = p.position.x, p.position.y, yaw_from_quaternion(p.orientation)
        if self.origin is None:
            self.origin = (x, y, yaw)
            self.get_logger().info(
                f'Origin set to x={x:.3f} y={y:.3f} yaw={math.degrees(yaw):.1f} deg')
        x0, y0, yaw0 = self.origin
        c, s = math.cos(yaw0), math.sin(yaw0)
        dx, dy = x - x0, y - y0
        return (c * dx + s * dy, -s * dx + c * dy, wrap_angle(yaw - yaw0))

    def scan_for(self, stamp):
        """The scan /vlio_odom was computed from (same stamp), else the newest one."""
        for scan_stamp, scan in reversed(self.scans):
            if abs(scan_stamp - stamp) < 1e-6:
                return scan
        return self.scans[-1][1]

    def next_goal(self):
        self.goal_index += 1
        self.state_machine = StateMachine(State.GO_TO_GOAL)
        if self.goal_index < len(self.waypoints):
            x, y = self.waypoints[self.goal_index]
            self.get_logger().info(f'Heading to waypoint {self.goal_index + 1} ({x:g}, {y:g})')
        else:
            self.get_logger().info('All waypoints reached; done')

    def timeout_callback(self):
        """Stop the robot once if no /vlio_odom has arrived for POSE_TIMEOUT."""
        if self.last_pose_time is None:
            return
        age = self.get_clock().now() - self.last_pose_time
        if age > Duration(seconds=POSE_TIMEOUT):
            self.get_logger().warn('No /vlio_odom, stopping')
            self.last_pose_time = None
            self.publish_cmd(0.0, 0.0)

    def publish_cmd(self, linear_x, angular_z):
        cmd = Twist()
        cmd.linear.x = float(linear_x)
        cmd.angular.z = float(angular_z)
        self.cmd_vel_pub.publish(cmd)


def main(args=None):
    # Handle Ctrl-C ourselves so the context is still valid to send a final stop.
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = GoToGoal()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_cmd(0.0, 0.0)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
