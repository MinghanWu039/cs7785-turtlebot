#!/usr/bin/env python3

import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from rclpy.time import Time

from action_msgs.msg import GoalStatus, GoalStatusArray
from nav_msgs.msg import Path
from sensor_msgs.msg import LaserScan
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from turtlebot3_msgs.srv import Sound
from turtlebot3_recovery_lab.srv import DivertRobot

MAP_FRAME = 'map'
ROBOT_FRAME = 'base_footprint'

PATH_MIN_DISTANCE = 1.0  # m: try 1 picks the closest path pose at least this far away

# Try 2: sample points around the robot and keep one with a clear straight line to it
SAMPLE_COUNT = 20
SAMPLE_MIN_DISTANCE = 0.5  # m: sampled points are at least this far from the robot
SAMPLE_MAX_DISTANCE = 1.5  # m: and at most this far (well inside the lidar's range)
SAMPLE_CLEARANCE = 0.25  # m: no scan point may be this close to the robot->point segment
SCAN_MAX_AGE = 1.0  # s: ignore a scan older than this (time since it was received)

FIXED_RECOVERY_POINT = (0.0, 0.0)  # map frame: try 3 (TODO: fill in)

# Matches the action server's status topic: one latched message.
LATCHED_QOS = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


def yaw_from_quaternion(q):
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


class StudentRecoveryServer(Node):
    def __init__(self):
        super().__init__('student_recovery_server')

        # Robot pose comes from TF (map -> base_footprint)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Latest global plan to the goal, published by planner_server
        self.path = None
        self.create_subscription(Path, '/plan', self.plan_callback, 1)

        # Track the active NavigateToPose goal so the tries restart per goal
        self.active_goal_id = None
        self.create_subscription(
            GoalStatusArray,
            '/navigate_to_pose/_action/status',
            self.status_callback,
            LATCHED_QOS,
        )

        # Latest lidar scan (best-effort QoS, matching the TurtleBot3 driver)
        self.scan = None
        self.scan_received = None
        self.rng = np.random.default_rng()
        self.create_subscription(
            LaserScan, '/scan', self.scan_callback, qos_profile_sensor_data
        )

        # Recovery strategies, tried in this order across recovery calls
        self.strategies = [
            ('closest path pose >= 1.0 m away', self.path_waypoint),
            ('sampled free point >= 0.5 m away', self.sampled_waypoint),
            ('fixed recovery point', self.fixed_waypoint),
        ]
        self.next_strategy = 0
        self.last_recovery_xy = None

        # TurtleBot3 OpenCR buzzer service client
        self.sound_client = self.create_client(
            Sound,
            '/sound'
        )

        # Recovery service called by the C++ Behavior Tree bridge
        self.srv = self.create_service(
            DivertRobot,
            '/divert_robot',
            self.handle_recovery
        )

        self.get_logger().info(
            'Recovery Python Server listening on /divert_robot'
        )

        if self.sound_client.service_is_ready():
            self.get_logger().info(
                'TurtleBot3 sound service available'
            )
        else:
            self.get_logger().warn(
                'TurtleBot3 sound service not currently available'
            )

    def beep(self):
        """Trigger the TurtleBot3 error buzzer asynchronously."""

        if not self.sound_client.service_is_ready():
            self.get_logger().warn(
                'Cannot beep: /sound service unavailable'
            )
            return

        sound_request = Sound.Request()

        # TurtleBot3 sound commands:
        # 0 = OFF
        # 1 = ON
        # 2 = LOW_BATTERY
        # 3 = ERROR
        # 4 = BUTTON1
        # 5 = BUTTON2
        sound_request.value = 3

        try:
            future = self.sound_client.call_async(sound_request)
            future.add_done_callback(self.sound_response_callback)

            self.get_logger().info(
                'Sent ERROR sound request to TurtleBot3'
            )

        except Exception as exc:
            self.get_logger().error(
                f'Failed to send sound request: {exc}'
            )

    def sound_response_callback(self, future):
        """Process the asynchronous sound service response."""

        try:
            result = future.result()

            if result.success:
                self.get_logger().info(
                    'TurtleBot3 buzzer activated successfully'
                )
            else:
                self.get_logger().warn(
                    f'TurtleBot3 buzzer failed: {result.message}'
                )

        except Exception as exc:
            self.get_logger().error(
                f'Sound service call failed: {exc}'
            )

    # --------------------------------------------------
    # Subscriptions
    # --------------------------------------------------

    def plan_callback(self, msg):
        """Keep the latest plan to the goal, ignoring our own recovery plans."""

        if not msg.poses:
            return

        # The recovery branch's ComputePathToPose also publishes on /plan.
        # Those plans end at the recovery point we returned; skip them so
        # self.path stays the plan toward the real goal.
        if self.last_recovery_xy is not None:
            end = msg.poses[-1].pose.position
            if math.hypot(end.x - self.last_recovery_xy[0],
                          end.y - self.last_recovery_xy[1]) < 0.05:
                return

        self.path = msg

    def status_callback(self, msg):
        """Restart the recovery sequence whenever a new goal starts executing."""

        for status in msg.status_list:
            if status.status != GoalStatus.STATUS_EXECUTING:
                continue

            goal_id = bytes(status.goal_info.goal_id.uuid)
            if goal_id != self.active_goal_id:
                self.active_goal_id = goal_id
                self.next_strategy = 0
                self.last_recovery_xy = None
                self.path = None
                self.get_logger().info(
                    'New navigation goal: recovery sequence reset'
                )

    def scan_callback(self, msg):
        # Store the receive time, not msg.header.stamp: the robot's and the
        # laptop's clocks may disagree.
        self.scan = msg
        self.scan_received = self.get_clock().now()

    # --------------------------------------------------
    # Recovery strategies: each returns (x, y, yaw) or None
    # --------------------------------------------------

    def get_robot_pose(self, frame=ROBOT_FRAME):
        """Return the (x, y, yaw) of `frame` in the map frame, or None."""

        try:
            # Time() = latest available transform; no waiting, since the
            # BT bridge only gives this service a few milliseconds.
            tf = self.tf_buffer.lookup_transform(
                MAP_FRAME, frame, Time()
            )
        except TransformException as exc:
            self.get_logger().warn(
                f'Cannot get pose ({MAP_FRAME} -> {frame}): {exc}'
            )
            return None

        t = tf.transform.translation
        return t.x, t.y, yaw_from_quaternion(tf.transform.rotation)

    def path_waypoint(self, robot):
        """Try 1: the closest pose ahead on the plan that is >= 1.0 m away."""

        if robot is None or self.path is None:
            return None

        rx, ry, _ = robot
        poses = self.path.poses
        dists = [
            math.hypot(p.pose.position.x - rx, p.pose.position.y - ry)
            for p in poses
        ]

        # Only look ahead of the robot: start at the path pose nearest to it,
        # so poses the robot already drove past are not picked.
        start = min(range(len(dists)), key=dists.__getitem__)
        candidates = [
            i for i in range(start, len(poses))
            if dists[i] >= PATH_MIN_DISTANCE
        ]
        if not candidates:
            return None

        best = min(candidates, key=dists.__getitem__)
        pose = poses[best].pose
        return (
            pose.position.x,
            pose.position.y,
            yaw_from_quaternion(pose.orientation),
        )

    def scan_points(self):
        """Return the latest scan as an (N, 2) array in the lidar frame, or None."""

        if self.scan is None:
            return None

        age = (self.get_clock().now() - self.scan_received).nanoseconds * 1e-9
        if age > SCAN_MAX_AGE:
            self.get_logger().warn(f'Latest scan is {age:.1f} s old')
            return None

        s = self.scan
        ranges = np.asarray(s.ranges, dtype=float)
        angles = s.angle_min + s.angle_increment * np.arange(len(ranges))
        valid = (
            np.isfinite(ranges)
            & (ranges >= s.range_min)
            & (ranges <= s.range_max)
        )
        r, a = ranges[valid], angles[valid]
        return np.column_stack((r * np.cos(a), r * np.sin(a)))

    def sampled_waypoint(self, robot):
        """
        Try 2: sample SAMPLE_COUNT points 0.5-1.5 m around the robot and keep
        those whose straight segment from the robot stays SAMPLE_CLEARANCE
        away from every scan point. Return the one with the most clearance.

        Everything is computed in the lidar frame (the scan's own frame),
        with the lidar at the origin, then converted to the map frame.
        """

        points = self.scan_points()
        if points is None:
            return None

        lidar = self.get_robot_pose(self.scan.header.frame_id)
        if lidar is None:
            return None

        # Sample uniformly over the area of the ring between the two radii
        dist = np.sqrt(self.rng.uniform(
            SAMPLE_MIN_DISTANCE ** 2, SAMPLE_MAX_DISTANCE ** 2, SAMPLE_COUNT
        ))
        theta = self.rng.uniform(-math.pi, math.pi, SAMPLE_COUNT)
        samples = np.column_stack((dist * np.cos(theta), dist * np.sin(theta)))

        if len(points) == 0:
            # Nothing within lidar range: every sample is clear
            clearance = np.full(SAMPLE_COUNT, np.inf)
        else:
            # Distance from each scan point to each segment origin -> sample,
            # shape (samples, points): project onto the segment, clamp to it.
            t = (samples @ points.T) / (dist ** 2)[:, None]
            t = np.clip(t, 0.0, 1.0)
            closest = t[:, :, None] * samples[:, None, :]
            gaps = np.linalg.norm(points[None, :, :] - closest, axis=2)
            clearance = gaps.min(axis=1)

        free = np.flatnonzero(clearance >= SAMPLE_CLEARANCE)
        self.get_logger().info(
            f'{len(free)}/{SAMPLE_COUNT} sampled points have a clear path'
        )
        if len(free) == 0:
            return None

        best = free[np.argmax(clearance[free])]
        sx, sy = samples[best]

        # Lidar frame -> map frame
        lx, ly, lyaw = lidar
        c, s = math.cos(lyaw), math.sin(lyaw)
        x = lx + c * sx - s * sy
        y = ly + s * sx + c * sy
        heading = lyaw + math.atan2(sy, sx)  # face the direction of travel
        return float(x), float(y), heading

    def fixed_waypoint(self, robot):
        """Try 3: a fixed recovery point in the map frame."""

        return FIXED_RECOVERY_POINT[0], FIXED_RECOVERY_POINT[1], 0.0

    def handle_recovery(self, request, response):
        """
        Called when Nav2 triggers the custom recovery behavior.

        1. Trigger the TurtleBot3 buzzer.
        2. Compute a recovery waypoint.
        3. Return the waypoint to the C++ BT bridge if one is available.
        4. Trigger a different Turtlebot3 buzzer if successful.
        """

        self.get_logger().warn(
            'Nav2 stalled! Sounding alarm and calculating diversion route...'
        )

        # --------------------------------------------------
        # STEP 1: Trigger the TurtleBot3 buzzer
        # --------------------------------------------------

        self.beep()

        # --------------------------------------------------
        # STEP 2: Compute recovery waypoint
        # --------------------------------------------------

        # Each recovery call moves one step down the strategy list:
        #   call 1: closest path pose >= 1.0 m away
        #   call 2: sampled point with a clear line from the robot, >= 0.5 m away
        #   call 3+: fixed recovery point
        # If a strategy cannot produce a point (no path, no TF), fall through
        # to the next one within the same call.
        robot = self.get_robot_pose()
        waypoint = None

        for index in range(self.next_strategy, len(self.strategies)):
            name, strategy = self.strategies[index]
            waypoint = strategy(robot)
            if waypoint is not None:
                self.get_logger().info(f'Recovery strategy: {name}')
                self.next_strategy = min(index + 1, len(self.strategies) - 1)
                break
            self.get_logger().warn(f'Recovery strategy unavailable: {name}')

        # --------------------------------------------------
        # STEP 3: Return recovery waypoint
        # --------------------------------------------------

        if waypoint is None:
            self.get_logger().error('No recovery waypoint available')
            response.success = False
            return response

        recovery_x, recovery_y, recovery_yaw = waypoint
        self.last_recovery_xy = (recovery_x, recovery_y)

        response.recovery_pose.header.frame_id = MAP_FRAME
        response.recovery_pose.header.stamp = (
            self.get_clock().now().to_msg()
        )

        response.recovery_pose.pose.position.x = float(recovery_x)
        response.recovery_pose.pose.position.y = float(recovery_y)
        response.recovery_pose.pose.position.z = 0.0

        response.recovery_pose.pose.orientation.x = 0.0
        response.recovery_pose.pose.orientation.y = 0.0
        response.recovery_pose.pose.orientation.z = math.sin(recovery_yaw / 2.0)
        response.recovery_pose.pose.orientation.w = math.cos(recovery_yaw / 2.0)

        response.success = True

        self.get_logger().info(
            f'Recovery waypoint generated: '
            f'({recovery_x:.2f}, {recovery_y:.2f})'
        )

        # ---------------------------------------------------
        # STEP 4: Trigger a different TurtleBot3 buzzer if a
        # recovery point was found
        # ---------------------------------------------------

        return response


def main(args=None):
    rclpy.init(args=args)

    node = StudentRecoveryServer()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()