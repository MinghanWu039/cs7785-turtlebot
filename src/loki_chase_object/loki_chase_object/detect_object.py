import cv2
from loki_object_follower.find_object import FindObject
from loki_object_follower.find_object import NO_OBJECT_POINT
from loki_object_follower.find_object import normalize_point
from loki_object_follower_msgs.msg import ImagePoint
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from std_msgs.msg import Float32


def pixel_bearing(frame, point, horizontal_fov):
    """
    Convert a raw-pixel object Point (as returned by FindObject.detect_object)
    to a bearing in radians.

    Follows REP-103: a positive bearing means the object is to the left of
    the camera's optical axis (matching rotate_robot's +angular.z = turn left).
    """
    width = frame.shape[1]
    x_norm = (point.x - width / 2.0) / (width / 2.0)  # [-1, 1], +right
    return float(-x_norm * (horizontal_fov / 2.0))


class DetectObject(FindObject):
    """
    Find the tracked object in the camera image and publish its bearing.

    Reuses loki_object_follower's FindObject detection algorithm (lab 2)
    unchanged - background subtraction, hue tracking, circle fit - and adds
    a bearing in radians published on /object_bearing, computed from the
    detected pixel column and horizontal_fov. Still publishes the normalized
    Point on /object and the ImagePoint/mask debug topics, so lab 2's
    debug_viz tooling keeps working.
    """

    def __init__(self):
        super().__init__()
        # TODO: replace with the camera's actual horizontal FOV.
        self.horizontal_fov = self.declare_parameter(
            'horizontal_fov', 1.0862).value  # rad (~62.2 deg, Pi Camera v2 default)
        self.bearing_pub = self.create_publisher(Float32, '/object_bearing', 10)

    def image_callback(self, msg):
        frame = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            self.get_logger().warn('Could not decode compressed image')
            return

        prev_status = self.state.status
        self.foreground_mask = None
        point = self.detect_object(frame)
        if point is not None:
            self.object_pub.publish(normalize_point(frame, point))
            self.bearing_pub.publish(
                Float32(data=pixel_bearing(frame, point, self.horizontal_fov)))
        self.debug_viz_pub.publish(
            ImagePoint(image=msg, point=NO_OBJECT_POINT if point is None else point))
        self.publish_mask(msg.header, frame.shape[:2])

        if prev_status != self.state.status:
            self.get_logger().info(f'Status: {self.state.status}')
        if self.display:
            cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = DetectObject()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node.display:
            cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
