import socket
import struct
import threading

from loki_object_follower_msgs.msg import ImagePoint
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage

# Wire format, per frame (big-endian): uint32 length, then `length` bytes made of
# one kind byte followed by the payload:
#   IMAGE: float32 x, float32 y, float32 radius, then the JPEG image bytes.
#   MASK:  the PNG foreground mask bytes.
HEADER = struct.Struct('>I')
POINT = struct.Struct('>fff')
IMAGE = b'I'
MASK = b'M'


class DebugVizServer(Node):
    """Forward /debug_viz and /debug_mask over TCP so a machine without ROS/DDS can show them."""

    def __init__(self):
        super().__init__('debug_viz_server')
        # Bind to localhost by default; use an ssh tunnel or set host to 0.0.0.0.
        host = self.declare_parameter('host', '127.0.0.1').value
        port = self.declare_parameter('port', 5555).value

        self.cond = threading.Condition()
        self.seq = 0
        self.latest = {}  # kind -> (seq, payload); only the newest frame of each kind is kept

        self.create_subscription(
            ImagePoint, '/debug_viz', self.viz_callback, qos_profile_sensor_data)
        self.create_subscription(
            CompressedImage, '/debug_mask', self.mask_callback, qos_profile_sensor_data)

        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((host, port))
        self.server.listen()
        self.get_logger().info(f'Serving /debug_viz and /debug_mask on {host}:{port}')
        threading.Thread(target=self.accept_loop, daemon=True).start()

    def viz_callback(self, msg):
        body = POINT.pack(msg.point.x, msg.point.y, msg.point.z) + bytes(msg.image.data)
        self.store(IMAGE, body)

    def mask_callback(self, msg):
        self.store(MASK, bytes(msg.data))

    def store(self, kind, body):
        body = kind + body
        with self.cond:
            self.seq += 1
            self.latest[kind] = (self.seq, HEADER.pack(len(body)) + body)
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
        last_seq = 0
        try:
            while True:
                with self.cond:
                    self.cond.wait_for(lambda: self.seq != last_seq, timeout=1.0)
                    if self.seq == last_seq:
                        continue
                    # A slow client only ever gets the newest frame of each kind.
                    fresh = sorted(v for v in self.latest.values() if v[0] > last_seq)
                    last_seq = self.seq
                for _, payload in fresh:
                    conn.sendall(payload)
        except OSError:
            pass
        finally:
            conn.close()
            self.get_logger().info(f'Client disconnected: {addr[0]}:{addr[1]}')

    def close(self):
        self.server.close()


def main(args=None):
    rclpy.init(args=args)
    node = DebugVizServer()
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
