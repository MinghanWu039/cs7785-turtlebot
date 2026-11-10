import math
import os
import re

from action_msgs.msg import GoalStatus, GoalStatusArray
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav2_msgs.action import NavigateToPose
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.signals import SignalHandlerOptions

FEEDBACK_LOG_PERIOD = 2.0  # s between progress log lines

STATUS_NAMES = {
    GoalStatus.STATUS_SUCCEEDED: 'SUCCEEDED',
    GoalStatus.STATUS_ABORTED: 'ABORTED',
    GoalStatus.STATUS_CANCELED: 'CANCELED',
}

# Matches the publishers in AMCL and in the action server's status topic: one latched message.
LATCHED_QOS = QoSProfile(
    depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)


def load_waypoints(path):
    """
    Read 'x y [yaw_deg]' per line, separated by spaces or commas.

    Blank lines and lines starting with '#' are skipped. A missing yaw is returned as None.
    """
    waypoints = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            values = [float(v) for v in re.split(r'[,\s]+', line)]
            yaw = math.radians(values[2]) if len(values) > 2 else None
            waypoints.append((values[0], values[1], yaw))
    return waypoints


def goal_key(goal_id):
    return bytes(goal_id.uuid)


class WaypointPublisher(Node):
    """
    Send the waypoints to Nav2 one at a time.

    Waits until AMCL has a pose (set the initial pose in RViz) and bt_navigator is listening,
    then publishes the first waypoint as a PoseStamped on /goal_pose. bt_navigator turns it
    into a NavigateToPose goal. The node follows that goal on
    /navigate_to_pose/_action/feedback (pose, distance remaining) and
    /navigate_to_pose/_action/status. When the goal SUCCEEDS, it waits goal_pause seconds
    and publishes the next waypoint. An ABORTED or CANCELED goal is resent up to
    max_retries times, then skipped.
    """

    def __init__(self):
        super().__init__('waypoint_publisher')
        # Empty means the waypoints.txt installed with this package.
        waypoints_file = self.declare_parameter('waypoints_file', '').value or os.path.join(
            get_package_share_directory('loki_nav2_waypoints'), 'waypoints.txt')
        self.goal_pause = self.declare_parameter('goal_pause', 2.0).value  # s
        self.max_retries = self.declare_parameter('max_retries', 2).value
        self.frame_id = self.declare_parameter('frame_id', 'map').value

        self.waypoints = load_waypoints(waypoints_file)
        self.index = 0  # waypoint being driven to, or to send next
        self.retries = 0
        self.robot_xy = None  # latest map-frame position, from AMCL or Nav2 feedback
        self.seen_goals = set()  # goal ids that already existed before our current goal
        self.goal_sent = False  # current waypoint published and not finished yet
        self.goal_id = None  # Nav2's id for the current waypoint, once it appears
        self.next_send_time = None  # when to publish the next waypoint (after the pause)
        self.done = False

        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.create_subscription(
            PoseWithCovarianceStamped, '/amcl_pose', self.amcl_callback, LATCHED_QOS)
        self.create_subscription(
            NavigateToPose.Impl.FeedbackMessage, '/navigate_to_pose/_action/feedback',
            self.feedback_callback, 10)
        self.create_subscription(
            GoalStatusArray, '/navigate_to_pose/_action/status', self.status_callback,
            LATCHED_QOS)
        self.create_timer(0.5, self.timer_callback)

        listed = ', '.join(f'({x:g}, {y:g})' for x, y, _ in self.waypoints)
        self.get_logger().info(f'{len(self.waypoints)} waypoints from {waypoints_file}: {listed}')
        self.get_logger().info('Waiting for an AMCL pose (set 2D Pose Estimate in RViz)...')

    def amcl_callback(self, msg):
        first = self.robot_xy is None
        self.robot_xy = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        if first:
            self.get_logger().info(
                f'AMCL pose received at ({self.robot_xy[0]:.2f}, {self.robot_xy[1]:.2f})')

    def timer_callback(self):
        if self.done or self.goal_sent or self.robot_xy is None:
            return
        if self.next_send_time is not None and self.get_clock().now() < self.next_send_time:
            return
        if self.goal_pub.get_subscription_count() == 0:
            self.get_logger().info(
                'Waiting for bt_navigator to subscribe to /goal_pose...',
                throttle_duration_sec=5.0)
            return
        self.send_goal()

    def send_goal(self):
        x, y, yaw = self.waypoints[self.index]
        if yaw is None:
            # Face the direction of travel from the previous waypoint (or the robot).
            px, py = self.waypoints[self.index - 1][:2] if self.index > 0 else self.robot_xy
            yaw = math.atan2(y - py, x - px)

        msg = PoseStamped()
        msg.header.frame_id = self.frame_id
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.orientation.z = math.sin(yaw / 2.0)
        msg.pose.orientation.w = math.cos(yaw / 2.0)
        self.goal_pub.publish(msg)

        self.goal_sent = True
        self.goal_id = None
        self.next_send_time = None
        retry = f' (retry {self.retries})' if self.retries else ''
        self.get_logger().info(
            f'Sent waypoint {self.index + 1}/{len(self.waypoints)}: '
            f'({x:.2f}, {y:.2f}, {math.degrees(yaw):.0f} deg){retry}')

    def claim(self, goal_id):
        """Return True if goal_id is the goal for the current waypoint."""
        key = goal_key(goal_id)
        if self.goal_id is not None:
            return key == self.goal_id
        if key in self.seen_goals:
            return False
        if not self.goal_sent:
            self.seen_goals.add(key)  # a goal from before ours, e.g. one sent from RViz
            return False
        self.goal_id = key  # the first new goal after we published is ours
        return True

    def feedback_callback(self, msg):
        if not self.claim(msg.goal_id):
            return
        fb = msg.feedback
        pos = fb.current_pose.pose.position
        self.robot_xy = (pos.x, pos.y)
        self.get_logger().info(
            f'Waypoint {self.index + 1}: at ({pos.x:.2f}, {pos.y:.2f}), '
            f'{fb.distance_remaining:.2f} m to go, {fb.number_of_recoveries} recoveries',
            throttle_duration_sec=FEEDBACK_LOG_PERIOD)

    def status_callback(self, msg):
        for status in msg.status_list:
            if not self.claim(status.goal_info.goal_id):
                continue
            if status.status in STATUS_NAMES:
                self.finish_goal(status.status)
                return

    def finish_goal(self, status):
        self.seen_goals.add(self.goal_id)
        self.goal_sent = False
        self.goal_id = None
        name = STATUS_NAMES[status]

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info(f'Reached waypoint {self.index + 1}')
        elif self.retries < self.max_retries:
            self.retries += 1
            self.get_logger().warn(f'Waypoint {self.index + 1} {name}; resending')
            self.next_send_time = self.get_clock().now() + Duration(seconds=1.0)
            return
        else:
            self.get_logger().error(
                f'Waypoint {self.index + 1} {name} after {self.retries} retries; skipping it')

        self.index += 1
        self.retries = 0
        if self.index >= len(self.waypoints):
            self.done = True
            self.get_logger().info('All waypoints visited')
            return
        self.next_send_time = self.get_clock().now() + Duration(seconds=self.goal_pause)


def main(args=None):
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = WaypointPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
