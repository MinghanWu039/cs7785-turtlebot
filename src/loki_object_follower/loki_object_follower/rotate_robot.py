from geometry_msgs.msg import Point
from geometry_msgs.msg import Twist
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions


DEADBAND = 0.1
TURN_SPEED = 0.3  # rad/s
OBJECT_TIMEOUT = 1.0  # s


def angular_velocity(x):
    """Return the turn rate for a normalized object x (+x right); +z turns left."""
    if x < -DEADBAND:
        return TURN_SPEED
    if x > DEADBAND:
        return -TURN_SPEED
    return 0.0


class RotateRobot(Node):
    """Rotate the robot in place to face the object published on /object."""

    def __init__(self):
        super().__init__('rotate_robot')
        self.object_sub = self.create_subscription(
            Point, '/object', self.object_callback, 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.last_object_time = None
        self.timer = self.create_timer(0.1, self.timer_callback)

    def object_callback(self, msg):
        if self.last_object_time is None:
            self.get_logger().info('Object found')
        self.last_object_time = self.get_clock().now()
        self.publish_cmd(angular_velocity(msg.x))

    def timer_callback(self):
        """Stop the robot once if no object has been seen for OBJECT_TIMEOUT."""
        if self.last_object_time is None:
            return
        age = self.get_clock().now() - self.last_object_time
        if age > Duration(seconds=OBJECT_TIMEOUT):
            self.get_logger().info('Object lost, stopping')
            self.last_object_time = None
            self.publish_cmd(0.0)

    def publish_cmd(self, angular_z):
        cmd = Twist()
        cmd.angular.z = angular_z
        self.cmd_vel_pub.publish(cmd)


def main(args=None):
    # Handle Ctrl-C ourselves so the context is still valid to send a final stop.
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = RotateRobot()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_cmd(0.0)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
