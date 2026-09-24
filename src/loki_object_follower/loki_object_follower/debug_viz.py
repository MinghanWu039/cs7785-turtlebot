import cv2
from loki_object_follower_msgs.msg import ImagePoint
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data


WINDOW = 'debug_viz'


class DebugViz(Node):
    """Show the image from /debug_viz with the detected object drawn on it."""

    def __init__(self):
        super().__init__('debug_viz')
        # Best-effort so a slow link drops frames instead of stalling the stream.
        self.create_subscription(
            ImagePoint, '/debug_viz', self.callback, qos_profile_sensor_data)
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    def callback(self, msg):
        frame = cv2.imdecode(np.frombuffer(msg.image.data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            self.get_logger().warn('Could not decode compressed image')
            return

        center = (int(msg.point.x), int(msg.point.y))
        radius = int(msg.point.z)
        cv2.circle(frame, center, radius, (0, 255, 0), 3)
        cv2.circle(frame, center, 4, (0, 0, 255), -1)
        cv2.putText(
            frame,
            f'({center[0]}, {center[1]}) r={radius}',
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        cv2.imshow(WINDOW, frame)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = DebugViz()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
