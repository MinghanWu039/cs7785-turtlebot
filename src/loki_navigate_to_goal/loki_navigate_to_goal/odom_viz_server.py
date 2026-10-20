from collections import deque
import math
import socket
import struct
import threading

from nav_msgs.msg import Odometry
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan

from loki_navigate_to_goal.geometry import (
    interpolate_pose, scan_to_points, stamp_seconds, transform, wrap_angle, yaw_from_quaternion)

# Wire format, per message (big-endian): uint32 length, then `length` bytes made of
# one kind byte followed by the payload:
#   POSE: float64 stamp seconds, x [m], y [m], yaw [rad].
#   MAP:  float64 stamp seconds, then float32 (x, y) pairs [m]: one scan in the world frame.
# On connect a client first receives every pose recorded so far and the kept map frames,
# then new ones as they arrive.
HEADER = struct.Struct('>I')
POSE = struct.Struct('>dddd')
STAMP = struct.Struct('>d')
POSE_KIND = b'P'
MAP_KIND = b'M'
POSE_BUFFER = 200  # recent poses kept for placing scans (~6 s of /odom)
STALE_SCAN = 1.0  # s; a scan still waiting for its pose after this long is dropped


def frame(kind, body):
    return HEADER.pack(len(body) + 1) + kind + body


class OdomVizServer(Node):
    """Forward an odometry trajectory and recent scans over TCP for a machine without ROS."""

    def __init__(self):
        super().__init__('odom_viz_server')
        # Bind to localhost by default; use an ssh tunnel or set host to 0.0.0.0.
        host = self.declare_parameter('host', '127.0.0.1').value
        port = self.declare_parameter('port', 5556).value
        odom_topic = self.declare_parameter('odom_topic', '/odom').value
        scan_topic = self.declare_parameter('scan_topic', '/scan').value
        # Report poses relative to the first one, so the run starts at (0, 0) facing +x.
        self.zero_start = self.declare_parameter('zero_start', True).value
        # Every map_period seconds one scan becomes a map frame; the last map_frames are kept.
        self.map_period = self.declare_parameter('map_period', 0.5).value
        map_frames = self.declare_parameter('map_frames', 10).value
        # Map points outside the map_size x map_size square around the origin are dropped.
        self.map_half = self.declare_parameter('map_size', 10.0).value / 2
        # base_scan sits 3.2 cm behind base_link on the burger (turtlebot3_burger.urdf).
        self.laser_x_offset = self.declare_parameter('laser_x_offset', -0.032).value

        self.origin = None  # (x0, y0, yaw0) of the first pose when zero_start is set
        self.poses = deque(maxlen=POSE_BUFFER)  # (stamp, x, y, yaw) as sent to clients
        self.pending = None  # (stamp, points in the robot frame) waiting for its pose
        self.last_map_stamp = None

        self.cond = threading.Condition()
        self.history = []  # every packed pose; clients need the whole trajectory
        self.maps = deque(maxlen=map_frames)  # (seq, packed map frame)
        self.map_seq = 0

        self.create_subscription(
            Odometry, odom_topic, self.odom_callback, qos_profile_sensor_data)
        self.create_subscription(
            LaserScan, scan_topic, self.scan_callback, qos_profile_sensor_data)

        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((host, port))
        self.server.listen()
        self.get_logger().info(f'Serving {odom_topic} and {scan_topic} on {host}:{port}')
        threading.Thread(target=self.accept_loop, daemon=True).start()

    def odom_callback(self, msg):
        pose = msg.pose.pose
        x, y = pose.position.x, pose.position.y
        yaw = yaw_from_quaternion(pose.orientation)

        if self.zero_start:
            if self.origin is None:
                self.origin = (x, y, yaw)
                self.get_logger().info(
                    f'Origin set to x={x:.3f} y={y:.3f} yaw={math.degrees(yaw):.1f} deg')
            x0, y0, yaw0 = self.origin
            dx, dy = x - x0, y - y0
            c, s = math.cos(yaw0), math.sin(yaw0)
            x, y = c * dx + s * dy, -s * dx + c * dy
            yaw = wrap_angle(yaw - yaw0)

        stamp = stamp_seconds(msg.header.stamp)
        self.poses.append((stamp, x, y, yaw))
        with self.cond:
            self.history.append(frame(POSE_KIND, POSE.pack(stamp, x, y, yaw)))
            self.cond.notify_all()
        self.place_pending_scan()

    def scan_callback(self, msg):
        stamp = stamp_seconds(msg.header.stamp)
        if self.pending is not None and stamp - self.pending[0] > STALE_SCAN:
            self.pending = None  # its pose never came
        if self.last_map_stamp is not None and stamp - self.last_map_stamp < self.map_period:
            return
        self.last_map_stamp = stamp
        self.pending = (stamp, scan_to_points(msg.ranges, msg.angle_min, msg.angle_increment,
                                              msg.range_min, msg.range_max, self.laser_x_offset))
        self.place_pending_scan()

    def place_pending_scan(self):
        """Turn the pending scan into a map frame once a pose at or after its stamp exists."""
        # /vlio_odom is published after ICP finishes, so the scan often arrives first.
        if self.pending is None or not self.poses or self.poses[-1][0] < self.pending[0]:
            return
        stamp, points = self.pending
        self.pending = None
        points = transform(interpolate_pose(self.poses, stamp), points)
        points = points[(np.abs(points) <= self.map_half).all(axis=1)]
        body = STAMP.pack(stamp) + points.astype('>f4').tobytes()
        with self.cond:
            self.map_seq += 1
            self.maps.append((self.map_seq, frame(MAP_KIND, body)))
            self.cond.notify_all()

    def accept_loop(self):
        while True:
            try:
                conn, addr = self.server.accept()
            except OSError:
                return
            self.get_logger().info(f'Client connected: {addr[0]}:{addr[1]}')
            threading.Thread(target=self.serve_client, args=(conn, addr), daemon=True).start()

    def serve_client(self, conn, addr):
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sent = 0
        map_seq = 0
        try:
            while True:
                with self.cond:
                    self.cond.wait_for(
                        lambda: len(self.history) > sent or self.map_seq > map_seq, timeout=1.0)
                    if len(self.history) == sent and self.map_seq == map_seq:
                        continue
                    # A slow client gets every pose it missed, but only the kept map frames.
                    batch = b''.join(self.history[sent:])
                    batch += b''.join(p for seq, p in self.maps if seq > map_seq)
                    sent, map_seq = len(self.history), self.map_seq
                conn.sendall(batch)
        except OSError:
            pass
        finally:
            conn.close()
            self.get_logger().info(f'Client disconnected: {addr[0]}:{addr[1]}')

    def close(self):
        self.server.close()


def main(args=None):
    rclpy.init(args=args)
    node = OdomVizServer()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.close()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
