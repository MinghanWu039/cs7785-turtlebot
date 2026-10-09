from geometry_msgs.msg import Twist
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from std_msgs.msg import Float32MultiArray


MIN_DT = 0.2  # s, minimum elapsed time between controller updates
SPIKE_RATIO = 10.0  # reject integral accumulation if |error| > this x |previous error|
OBJECT_TIMEOUT = 1.0  # s, stop if no /object_state has arrived for this long

ANGULAR_KP = 1.0  # rad/s per rad of angle error
ANGULAR_KI = 0.1  # rad/s per (rad*s) of accumulated angle error
ANGULAR_MAX_SPEED = 2.5  # rad/s

LINEAR_KP = 0.4  # m/s per m of distance error
LINEAR_KI = 0.05  # m/s per (m*s) of accumulated distance error
LINEAR_MAX_SPEED = 0.1  # m/s


class ChaseObject(Node):
    """
    Drive the robot to face the tracked object and hold a desired distance.

    Subscribes to /object_state ([angle (rad), distance (m)]) and publishes
    Twist velocity commands on /cmd_vel: an angular PI controller to face
    the object, and a linear PI controller to hold the desired distance.
    Controller updates are gated to run only on ticks spaced at least
    MIN_DT apart, using the actual measured elapsed time as dt. If no
    /object_state message arrives for OBJECT_TIMEOUT, the robot is stopped.
    """

    def __init__(self):
        super().__init__('chase_object')
        self.desired_distance = self.declare_parameter('desired_distance', 0.5).value  # m

        self.angular_pid = PIDController(ANGULAR_KP, ANGULAR_KI, ANGULAR_MAX_SPEED)
        self.linear_pid = PIDController(LINEAR_KP, LINEAR_KI, LINEAR_MAX_SPEED)
        self.last_update_time = None
        self.last_object_time = None

        self.state_sub = self.create_subscription(
            Float32MultiArray, '/object_state', self.state_callback, 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.timeout_timer = self.create_timer(0.1, self.timeout_callback)

    def state_callback(self, msg):
        angle, distance = msg.data

        if self.last_object_time is None:
            self.get_logger().info('Object found')
        self.last_object_time = self.get_clock().now()

        now = self.get_clock().now()
        if self.last_update_time is None:
            # No baseline yet - record it and wait for the next tick instead
            # of computing a dt against node-startup time.
            self.last_update_time = now
            return

        dt = (now - self.last_update_time).nanoseconds / 1e9
        if dt < MIN_DT:
            return  # Not enough time elapsed since the last update; ignore this tick.
        self.last_update_time = now

        angular_error = angle  # +angle (object left) -> +angular.z turns left
        linear_error = distance - self.desired_distance  # too far -> drive forward

        angular_z = self.angular_pid.update(angular_error, dt)
        linear_x = self.linear_pid.update(linear_error, dt)

        self.publish_cmd(linear_x, angular_z)

    def timeout_callback(self):
        """Stop the robot once if no /object_state has arrived for OBJECT_TIMEOUT."""
        if self.last_object_time is None:
            return
        age = self.get_clock().now() - self.last_object_time
        if age > Duration(seconds=OBJECT_TIMEOUT):
            self.get_logger().info('Object lost, stopping')
            self.last_object_time = None
            # Restart the dt baseline too, so the controllers don't see a
            # huge dt spanning the gap once the object is seen again.
            self.last_update_time = None
            self.publish_cmd(0.0, 0.0)

    def publish_cmd(self, linear_x, angular_z):
        cmd = Twist()
        cmd.linear.x = linear_x
        cmd.angular.z = angular_z
        self.cmd_vel_pub.publish(cmd)


class PIDController:
    """
    A PI controller with a clamped integral accumulator.

    The integral accumulator is clamped to +/- output_limit / ki, so the
    integral term alone can never drive the output past the controller's
    saturation limit (simple anti-windup). If the magnitude of the current
    error is more than SPIKE_RATIO times the magnitude of the previous
    error, the integral accumulation step is skipped for this update
    (treated as an outlier) - the proportional term is still computed
    normally from the current error.
    """

    def __init__(self, kp, ki, output_limit):
        self.kp = kp
        self.ki = ki
        self.output_limit = output_limit
        self.integral_limit = output_limit / ki if ki != 0.0 else float('inf')
        self.integral = 0.0
        self.previous_error = None

    def update(self, error, dt):
        if self.previous_error is None or abs(error) <= SPIKE_RATIO * abs(self.previous_error):
            self.integral += error * dt
            self.integral = max(-self.integral_limit, min(self.integral_limit, self.integral))
        self.previous_error = error

        output = self.kp * error + self.ki * self.integral
        return max(-self.output_limit, min(self.output_limit, output))


def main(args=None):
    # Handle Ctrl-C ourselves so the context is still valid to send a final stop.
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = ChaseObject()
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
