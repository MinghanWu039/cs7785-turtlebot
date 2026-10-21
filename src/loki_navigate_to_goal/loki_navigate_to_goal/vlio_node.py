from collections import deque
import math

from nav_msgs.msg import Odometry
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan

from loki_navigate_to_goal.geometry import (
    compose, interpolate_pose, inverse, scan_to_points, stamp_seconds, transform, wrap_angle,
    yaw_from_quaternion)

ODOM_BUFFER = 200  # /odom messages kept for looking up the pose at a scan stamp (~6 s)


def icp(source, target, init, iterations=10, max_dist=0.5, min_matches=20):
    """
    Point-to-point ICP: find the pose T with transform(T, source) best overlapping target.

    Correspondences farther apart than max_dist are dropped in every iteration, including
    the first. Returns (T, matches, mean_error, ok); ok is False if an iteration had fewer
    than min_matches correspondences, in which case T is the last good estimate.
    """
    pose = tuple(init)
    matches, mean_error = 0, float('nan')
    for _ in range(iterations):
        moved = transform(pose, source)
        d2 = ((moved[:, None, :] - target[None, :, :]) ** 2).sum(axis=2)
        nearest = d2.argmin(axis=1)
        dist = np.sqrt(d2[np.arange(len(source)), nearest])
        keep = dist <= max_dist
        if keep.sum() < min_matches:
            return pose, int(keep.sum()), mean_error, False
        matches, mean_error = int(keep.sum()), float(dist[keep].mean())

        # Closed-form 2D rigid alignment of the original source points onto their matches.
        p, q = source[keep], target[nearest[keep]]
        p_mean, q_mean = p.mean(axis=0), q.mean(axis=0)
        h = (p - p_mean).T @ (q - q_mean)
        theta = math.atan2(h[0, 1] - h[1, 0], h[0, 0] + h[1, 1])
        c, s = math.cos(theta), math.sin(theta)
        t = q_mean - np.array([[c, -s], [s, c]]) @ p_mean
        pose = (float(t[0]), float(t[1]), theta)
    return pose, matches, mean_error, True


def estimate_normals(points, neighbors=5, max_spread=0.2):
    """
    Unit surface normal at each point, from the PCA of its nearest neighbours.

    Returns (normals, valid); a point is invalid if its neighbours reach farther than
    max_spread, where the local surface is too sparse to trust.
    """
    d2 = ((points[:, None, :] - points[None, :, :]) ** 2).sum(axis=2)
    idx = np.argsort(d2, axis=1)[:, :neighbors]
    valid = np.sqrt(d2[np.arange(len(points)), idx[:, -1]]) <= max_spread
    local = points[idx] - points[idx].mean(axis=1, keepdims=True)
    _, vectors = np.linalg.eigh(local.transpose(0, 2, 1) @ local)
    return vectors[:, :, 0], valid  # eigenvector of the smallest eigenvalue


def icp_point_to_line(source, target, init, iterations=10, max_dist=0.5, min_matches=20):
    """
    Point-to-line ICP: like icp(), but minimise each point's distance to the line through
    its match along the target's local surface, so points may slide along walls.

    Correspondences are gated on their point-to-point distance, as in icp(). Each
    iteration takes one Gauss-Newton step with the rotation linearised.
    """
    normals, has_normal = estimate_normals(target)
    pose = tuple(init)
    matches, mean_error = 0, float('nan')
    for _ in range(iterations):
        moved = transform(pose, source)
        d2 = ((moved[:, None, :] - target[None, :, :]) ** 2).sum(axis=2)
        nearest = d2.argmin(axis=1)
        dist = np.sqrt(d2[np.arange(len(source)), nearest])
        keep = (dist <= max_dist) & has_normal[nearest]
        if keep.sum() < min_matches:
            return pose, int(keep.sum()), mean_error, False

        # Residual r = n . (R p + t - q); its derivative in theta is n . (J R p), J = rot(90).
        rotated = moved[keep] - np.array(pose[:2])
        n, q = normals[nearest[keep]], target[nearest[keep]]
        r = (n * (moved[keep] - q)).sum(axis=1)
        a = np.column_stack([n, n[:, 1] * rotated[:, 0] - n[:, 0] * rotated[:, 1]])
        matches, mean_error = int(keep.sum()), float(np.abs(r).mean())
        # Light damping keeps the step bounded when the geometry is degenerate (a corridor).
        h = a.T @ a
        step = np.linalg.solve(h + 1e-6 * np.trace(h) * np.eye(3), -a.T @ r)
        pose = (pose[0] + step[0], pose[1] + step[1], wrap_angle(pose[2] + step[2]))
    return tuple(float(v) for v in pose), matches, mean_error, True


ICP_METHODS = {'point_to_point': icp, 'point_to_line': icp_point_to_line}


class VlioNode(Node):
    """
    Refine wheel odometry by ICP-aligning each /scan to a keyframe scan.

    Every scan is matched against the same keyframe until the robot has moved keyframe_dist
    or turned keyframe_angle from it, and only then becomes the new keyframe. Scan noise
    (the LDS-02's beam angles jitter by about 0.3 deg per revolution) then stays a bounded
    error instead of accumulating on every scan. keyframe_dist:=0.0 gives scan-to-scan ICP.
    """

    def __init__(self):
        super().__init__('vlio_node')
        odom_topic = self.declare_parameter('odom_topic', '/odom').value
        scan_topic = self.declare_parameter('scan_topic', '/scan').value
        output_topic = self.declare_parameter('output_topic', '/vlio_odom').value
        # use_icp:=false passes the odometry through unchanged, at the scan rate.
        self.use_icp = self.declare_parameter('use_icp', True).value
        self.iterations = self.declare_parameter('iterations', 10).value
        # point_to_line lets points slide along walls; point_to_point is the plain variant.
        method = self.declare_parameter('icp_method', 'point_to_line').value
        if method not in ICP_METHODS:
            raise ValueError(f'icp_method must be one of {sorted(ICP_METHODS)}, got {method!r}')
        self.icp = ICP_METHODS[method]
        self.max_dist = self.declare_parameter('max_correspondence_dist', 0.5).value
        self.min_matches = self.declare_parameter('min_matches', 20).value
        # base_scan sits 3.2 cm behind base_link on the burger (turtlebot3_burger.urdf).
        self.laser_x_offset = self.declare_parameter('laser_x_offset', -0.032).value
        self.keyframe_dist = self.declare_parameter('keyframe_dist', 0.2).value  # m
        self.keyframe_angle = math.radians(
            self.declare_parameter('keyframe_angle', 10.0).value)  # parameter in degrees

        self.odom = deque(maxlen=ODOM_BUFFER)  # (stamp, x, y, theta)
        self.last_pose = None  # pose at the latest scan, as published
        self.kf_pose = None  # pose of the keyframe scan
        self.kf_odom = None  # odometry pose at the keyframe scan
        self.kf_points = None

        self.create_subscription(
            Odometry, odom_topic, self.odom_callback, qos_profile_sensor_data)
        self.create_subscription(
            LaserScan, scan_topic, self.scan_callback, qos_profile_sensor_data)
        self.pub = self.create_publisher(Odometry, output_topic, 10)
        mode = (f'{method} ICP, new keyframe every {self.keyframe_dist:g} m or '
                f'{math.degrees(self.keyframe_angle):g} deg'
                if self.use_icp else 'ICP disabled, odometry only')
        self.get_logger().info(f'{mode} on {scan_topic} + {odom_topic} -> {output_topic}')

    def odom_callback(self, msg):
        pose = msg.pose.pose
        self.odom.append((stamp_seconds(msg.header.stamp), pose.position.x, pose.position.y,
                          yaw_from_quaternion(pose.orientation)))

    def scan_callback(self, msg):
        if not self.odom:
            return
        odom_now = interpolate_pose(self.odom, stamp_seconds(msg.header.stamp))
        points = scan_to_points(msg.ranges, msg.angle_min, msg.angle_increment,
                                msg.range_min, msg.range_max, self.laser_x_offset)

        if self.kf_pose is None:
            # Start in the odometry frame so /vlio_odom is directly comparable to /odom.
            self.last_pose = odom_now
            self.set_keyframe(points, odom_now)
        else:
            odom_from_kf = compose(inverse(self.kf_odom), odom_now)
            if self.use_icp:
                from_kf, ok = self.refine(points, odom_from_kf)
            else:
                from_kf, ok = odom_from_kf, True
            self.last_pose = compose(self.kf_pose, from_kf)
            moved = math.hypot(from_kf[0], from_kf[1]) > self.keyframe_dist
            turned = abs(from_kf[2]) > self.keyframe_angle
            # A failed match also starts a new keyframe, so the next scans overlap with it.
            if moved or turned or not ok:
                self.set_keyframe(points, odom_now)
        self.publish(msg.header.stamp)

    def set_keyframe(self, points, odom_now):
        self.kf_pose, self.kf_odom, self.kf_points = self.last_pose, odom_now, points

    def refine(self, points, odom_from_kf):
        """
        ICP-align points to the keyframe, starting from the odometry motion since it.

        Returns (motion since the keyframe, ok); not ok means too few correspondences, and
        the odometry motion is returned instead.
        """
        step, matches, error, ok = self.icp(points, self.kf_points, odom_from_kf,
                                            self.iterations, self.max_dist, self.min_matches)
        if not ok:
            # Too little overlap to trust the partial estimate; keep the odometry motion.
            self.get_logger().warn(
                f'ICP had only {matches} correspondences; using odometry for this scan',
                throttle_duration_sec=2.0)
            return odom_from_kf, False
        self.get_logger().debug(
            f'ICP {matches} matches, mean error {error:.3f} m, correction '
            f'dx={step[0] - odom_from_kf[0]:+.3f} dy={step[1] - odom_from_kf[1]:+.3f} '
            f'dth={math.degrees(wrap_angle(step[2] - odom_from_kf[2])):+.2f} deg')
        return step, True

    def publish(self, stamp):
        x, y, theta = self.last_pose
        out = Odometry()
        out.header.stamp = stamp
        out.header.frame_id = 'odom'
        out.child_frame_id = 'base_footprint'
        out.pose.pose.position.x = x
        out.pose.pose.position.y = y
        out.pose.pose.orientation.z = math.sin(theta / 2)
        out.pose.pose.orientation.w = math.cos(theta / 2)
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = VlioNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
