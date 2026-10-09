import math

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32
from std_msgs.msg import Float32MultiArray


RANGE_SEARCH_WINDOW = 5  # indices on each side of the bearing to search for a valid return

# Yaw angle (rad) of the LIDAR frame relative to the camera frame: the angle
# to add to a camera bearing to get the equivalent angle in the LIDAR's
# frame. TODO: measure/calibrate this on the robot; 0 assumes the two
# sensors' zero-bearing directions are physically aligned.
LIDAR_CAMERA_YAW_OFFSET = 0.0  # rad


def closest_valid_range(scan, index, window=RANGE_SEARCH_WINDOW):
    for offset in range(window + 1):
        for i in {index - offset, index + offset}:
            if not (0 <= i < len(scan.ranges)):
                continue
            r = scan.ranges[i]
            if math.isfinite(r) and scan.range_min <= r <= scan.range_max:
                return r
    return None


class GetObjectRange(Node):
    """
    Fuse the camera bearing with LIDAR to estimate the object's angle and distance.

    Subscribes to /object_bearing (camera, rad) and /scan (LIDAR), and
    publishes [angle (rad), distance (m)] on /object_state.

    TODO: look up the scan range at (or nearest) the camera bearing -
    accounting for the camera/LIDAR frame offset - and publish the fused
    angle + distance.
    """

    def __init__(self):
        super().__init__('get_object_range')
        self.bearing_sub = self.create_subscription(
            Float32, '/object_bearing', self.bearing_callback, 10)
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, qos_profile_sensor_data)
        self.state_pub = self.create_publisher(Float32MultiArray, '/object_state', 10)

        self.last_bearing = None
        self.last_scan = None

    def bearing_callback(self, msg):
        self.last_bearing = msg.data
        self.publish_state()

    def scan_callback(self, msg):
        self.last_scan = msg
        self.publish_state()

    def publish_state(self):
        state = self.calculate_object_state()
        if state is not None:
            self.state_pub.publish(state)

    def calculate_object_state(self):
        """
        Calculate the object's angle and distance based on the last bearing and scan.

        Returns:
            Float32MultiArray: A message containing [angle (rad), distance (m)],
            or None if there's no bearing/scan yet, or no valid return near
            the bearing.
        """
        if self.last_bearing is None or self.last_scan is None:
            return None

        scan = self.last_scan
        angle = self.last_bearing

        # Convert the camera bearing into the LIDAR's frame before indexing,
        # then wrap it into the scan's [angle_min, angle_max] span.
        lidar_angle = angle + LIDAR_CAMERA_YAW_OFFSET
        span = scan.angle_max - scan.angle_min
        wrapped = (lidar_angle - scan.angle_min) % span + scan.angle_min
        index = round((wrapped - scan.angle_min) / scan.angle_increment)
        index = max(0, min(len(scan.ranges) - 1, index))

        distance = closest_valid_range(scan, index)
        if distance is None:
            return None

        # Publish the camera-frame bearing (not the LIDAR-frame angle) since
        # that's what chase_object steers on, relative to the robot's facing.
        return Float32MultiArray(data=[angle, distance])


def main(args=None):
    rclpy.init(args=args)
    node = GetObjectRange()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
