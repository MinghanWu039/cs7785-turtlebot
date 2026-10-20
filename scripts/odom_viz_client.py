#!/usr/bin/env python3
"""
Plot the robot's x-y trajectory and recent lidar scans streamed by the odom_viz_server node.

Needs only opencv-python and numpy, no ROS. Typical use over an ssh tunnel:

    ssh -N -L 5556:localhost:5556 <user>@<robot>
    python3 scripts/odom_viz_client.py --waypoints wayPoints.txt
"""

import argparse
from collections import deque
import math
import re
import socket
import struct
import time

import cv2
import numpy as np

# Must match src/loki_navigate_to_goal/loki_navigate_to_goal/odom_viz_server.py.
HEADER = struct.Struct('>I')
POSE = struct.Struct('>dddd')
STAMP = struct.Struct('>d')
POSE_KIND = b'P'
MAP_KIND = b'M'
MAP_FRAMES = 10  # the server keeps the same number

WINDOW = 'odometry'
SIZE = 800  # canvas width and height in pixels
MARGIN = 30  # pixels kept free around the plotted area
GOAL_RADII = (0.10, 0.15, 0.20)  # Lab 4 stopping tolerance for each goal, in order
REDRAW_PERIOD = 0.03


def load_waypoints(path):
    """Read 'x y' or 'x,y' per line; blank lines and lines starting with '#' are skipped."""
    waypoints = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            x, y = (float(v) for v in re.split(r'[,\s]+', line)[:2])
            waypoints.append((x, y))
    return waypoints


def read_messages(buf, poses, maps):
    """Consume every complete message in buf, appending to poses (list) and maps (deque)."""
    offset = 0
    while len(buf) - offset >= HEADER.size:
        (length,) = HEADER.unpack_from(buf, offset)
        if len(buf) - offset - HEADER.size < length:
            break
        body = bytes(buf[offset + HEADER.size:offset + HEADER.size + length])
        offset += HEADER.size + length
        kind, payload = body[:1], body[1:]
        if kind == POSE_KIND:
            poses.append(POSE.unpack(payload))
        elif kind == MAP_KIND:
            maps.append(np.frombuffer(payload[STAMP.size:], '>f4').reshape(-1, 2))
        else:
            raise ValueError(f'unknown message kind {kind!r}')
    del buf[:offset]


def draw(poses, maps, waypoints, canvas):
    """Render the map and trajectory on a fixed canvas x canvas metre view around (0, 0)."""
    image = np.full((SIZE, SIZE, 3), 255, np.uint8)
    scale = (SIZE - 2 * MARGIN) / canvas  # pixels per metre

    def to_px(x, y):
        return (int(round(SIZE / 2 + x * scale)), int(round(SIZE / 2 - y * scale)))

    # 1 m grid, labelled in metres, with the axes through the origin in black.
    half = math.floor(canvas / 2)
    for i in range(-half, half + 1):
        color = (0, 0, 0) if i == 0 else (225, 225, 225)
        px, py = to_px(i, i)
        cv2.line(image, (px, MARGIN), (px, SIZE - MARGIN), color, 1)
        cv2.line(image, (MARGIN, py), (SIZE - MARGIN, py), color, 1)
        cv2.putText(image, f'{i}', (px - 4, SIZE - MARGIN + 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (120, 120, 120), 1)
        cv2.putText(image, f'{i}', (MARGIN - 22, py + 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (120, 120, 120), 1)

    # Map frames, oldest first, so the newest scan is darkest and drawn on top.
    for age, points in enumerate(reversed(maps)):
        gray = int(40 + 170 * age / MAP_FRAMES)
        px = np.round(SIZE / 2 + points[:, 0] * scale).astype(int)
        py = np.round(SIZE / 2 - points[:, 1] * scale).astype(int)
        inside = (px >= 0) & (px < SIZE - 1) & (py >= 0) & (py < SIZE - 1)
        px, py = px[inside], py[inside]
        for dx in (0, 1):
            for dy in (0, 1):
                image[py + dy, px + dx] = np.minimum(image[py + dy, px + dx], gray)

    for i, (x, y) in enumerate(waypoints):
        radius = GOAL_RADII[min(i, len(GOAL_RADII) - 1)]
        cv2.circle(image, to_px(x, y), max(2, int(radius * scale)), (0, 140, 255), 2)
        cv2.putText(image, str(i + 1), to_px(x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 140, 255), 2)

    if poses:
        path = np.array([to_px(x, y) for _, x, y, _ in poses], np.int32)
        cv2.polylines(image, [path], False, (200, 80, 0), 2)
        cv2.circle(image, tuple(path[0]), 5, (0, 160, 0), -1)

        _, x, y, yaw = poses[-1]
        tip = to_px(x + 0.3 * math.cos(yaw), y + 0.3 * math.sin(yaw))
        cv2.arrowedLine(image, to_px(x, y), tip, (0, 0, 255), 2, tipLength=0.3)
        text = f'x={x:+.3f} m  y={y:+.3f} m  yaw={math.degrees(yaw):+.1f} deg'
    else:
        text = 'waiting for odometry'
    cv2.putText(image, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(image, f'{len(poses)} poses, {len(maps)} map frames, grid 1 m',
                (SIZE - 300, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 90, 90), 1)
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', type=int, default=5556)
    parser.add_argument('--waypoints', help='file of goal points, one "x y" per line')
    parser.add_argument('--canvas', type=float, default=10.0,
                        help='side of the square view around the start, in metres')
    args = parser.parse_args()
    waypoints = load_waypoints(args.waypoints) if args.waypoints else []

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    poses, maps = [], deque(maxlen=MAP_FRAMES)
    try:
        while True:
            cv2.imshow(WINDOW, draw(poses, maps, waypoints, args.canvas))
            try:
                sock = socket.create_connection((args.host, args.port), timeout=5)
            except OSError as exc:
                print(f'Cannot connect to {args.host}:{args.port} ({exc}); retrying')
                if cv2.waitKey(1000) == ord('q'):
                    return
                continue

            print(f'Connected to {args.host}:{args.port}')
            # The server resends the whole trajectory and the kept map on every connection.
            poses, maps = [], deque(maxlen=MAP_FRAMES)
            buf = bytearray()
            # A short timeout keeps the window responsive while the robot is idle.
            sock.settimeout(REDRAW_PERIOD)
            try:
                while True:
                    deadline = time.monotonic() + REDRAW_PERIOD
                    while time.monotonic() < deadline:
                        try:
                            chunk = sock.recv(65536)
                        except socket.timeout:
                            break
                        if not chunk:
                            raise ConnectionError('connection closed')
                        buf += chunk
                    read_messages(buf, poses, maps)
                    cv2.imshow(WINDOW, draw(poses, maps, waypoints, args.canvas))
                    if cv2.waitKey(1) == ord('q'):
                        return
            except (OSError, ConnectionError, ValueError, struct.error) as exc:
                print(f'Disconnected ({exc}); reconnecting')
                time.sleep(1)
            finally:
                sock.close()
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
