#!/usr/bin/env python3
"""
Display the detected object and foreground mask streamed by the robot's debug_viz_server node.

Needs only opencv-python and numpy, no ROS. Typical use over an ssh tunnel:

    ssh -N -L 5555:localhost:5555 <user>@<robot>
    python3 scripts/debug_viz_client.py
"""

import argparse
import socket
import struct
import time

import cv2
import numpy as np

# Must match src/loki_object_follower/loki_object_follower/debug_viz_server.py.
HEADER = struct.Struct('>I')
POINT = struct.Struct('>fff')
IMAGE = b'I'
MASK = b'M'
WINDOW = 'debug_viz'
MASK_WINDOW = 'foreground mask'


def recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError('connection closed')
        buf += chunk
    return bytes(buf)


def read_frame(sock):
    """
    Return the next frame as ('image', image, x, y, radius) or ('mask', mask).

    The decoded picture is None if it could not be decoded.
    """
    (length,) = HEADER.unpack(recv_exact(sock, HEADER.size))
    body = recv_exact(sock, length)
    kind, payload = body[:1], body[1:]
    if kind == IMAGE:
        x, y, radius = POINT.unpack(payload[:POINT.size])
        image = cv2.imdecode(np.frombuffer(payload[POINT.size:], np.uint8), cv2.IMREAD_COLOR)
        return 'image', image, x, y, radius
    if kind == MASK:
        mask = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_GRAYSCALE)
        return 'mask', mask
    raise ValueError(f'unknown frame kind {kind!r}')


def draw(image, x, y, radius):
    if radius < 0:  # the robot sends a negative radius when no object is detected
        cv2.putText(image, 'no object', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2)
        return
    center = (int(x), int(y))
    radius = int(radius)
    cv2.circle(image, center, radius, (0, 255, 0), 3)
    cv2.circle(image, center, 4, (0, 0, 255), -1)
    cv2.putText(
        image,
        f'({center[0]}, {center[1]}) r={radius}',
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', type=int, default=5555)
    args = parser.parse_args()

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.namedWindow(MASK_WINDOW, cv2.WINDOW_NORMAL)
    try:
        while True:
            try:
                sock = socket.create_connection((args.host, args.port), timeout=5)
            except OSError as exc:
                print(f'Cannot connect to {args.host}:{args.port} ({exc}); retrying')
                if cv2.waitKey(1000) == ord('q'):
                    return
                continue

            print(f'Connected to {args.host}:{args.port}')
            sock.settimeout(5)
            try:
                while True:
                    frame = read_frame(sock)
                    if frame[1] is None:
                        continue
                    if frame[0] == 'image':
                        _, image, x, y, radius = frame
                        draw(image, x, y, radius)
                        cv2.imshow(WINDOW, image)
                    else:
                        cv2.imshow(MASK_WINDOW, frame[1])
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
