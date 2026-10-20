"""SE(2) and lidar helpers shared by the navigate_to_goal nodes. Poses are (x, y, theta)."""

import math

import numpy as np


def yaw_from_quaternion(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def stamp_seconds(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def compose(a, b):
    """Return a ∘ b: pose b, expressed in a's frame, mapped into a's parent frame."""
    c, s = math.cos(a[2]), math.sin(a[2])
    return (a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], wrap_angle(a[2] + b[2]))


def inverse(a):
    c, s = math.cos(a[2]), math.sin(a[2])
    return (-c * a[0] - s * a[1], s * a[0] - c * a[1], wrap_angle(-a[2]))


def transform(pose, points):
    """Map an N x 2 array of points by pose."""
    c, s = math.cos(pose[2]), math.sin(pose[2])
    return points @ np.array([[c, s], [-s, c]]) + np.array(pose[:2])


def scan_to_points(ranges, angle_min, angle_increment, range_min, range_max, x_offset=0.0):
    """Convert a 1D range scan to an N x 2 array of points in the robot base frame."""
    ranges = np.asarray(ranges, dtype=float)
    angles = angle_min + angle_increment * np.arange(len(ranges))
    # The LDS reports a missing return as 0, which the range_min check removes.
    valid = np.isfinite(ranges) & (ranges >= range_min) & (ranges <= range_max)
    r, a = ranges[valid], angles[valid]
    return np.column_stack([r * np.cos(a) + x_offset, r * np.sin(a)])


def interpolate_pose(buffer, t):
    """
    Pose at time t from a time-ordered sequence of (stamp, x, y, theta).

    Interpolates linearly between neighbours and clamps to the buffered range.
    """
    if t <= buffer[0][0]:
        return tuple(buffer[0][1:])
    for before, after in zip(reversed(list(buffer)[:-1]), reversed(buffer)):
        if before[0] <= t:
            if t >= after[0]:
                return tuple(after[1:])
            f = (t - before[0]) / (after[0] - before[0])
            return (before[1] + f * (after[1] - before[1]),
                    before[2] + f * (after[2] - before[2]),
                    wrap_angle(before[3] + f * wrap_angle(after[3] - before[3])))
    return tuple(buffer[-1][1:])
