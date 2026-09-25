from enum import auto, Enum

import cv2
from geometry_msgs.msg import Point
from loki_object_follower_msgs.msg import ImagePoint
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage


CIRCLE_FIT_THRESHOLD = 0.5
DISTANCE_THRESHOLD = 500
NO_OBJECT_POINT = Point(x=-1.0, y=-1.0, z=-1.0)  # negative radius means "no object"


BACKGROUND_FRAMES = 5
FOREGROUND_THRESHOLD = 45


class Status(Enum):
    BACKGROUND = auto()
    OBJECT = auto()
    TRACKING = auto()


class State:

    def __init__(self):
        self.status = Status.BACKGROUND
        self.background_gray_buffer = []
        self.background_color_buffer = []
        self.background_gray = None
        self.background_color = None
        self.tracking_window = None
        self.roi_hist = None
        self.tracking_center = None
        self.hue_range = None


def full_foreground(frame):
    """Return a foreground mask that treats the whole frame as foreground."""
    return np.full(frame.shape[:2], 255, dtype=np.uint8)


def background_foreground(frame, background_gray, background_color,
                          threshold=FOREGROUND_THRESHOLD):
    """
    Return a foreground mask from differences against a stored background.

    Combines the grayscale intensity difference with a per-channel color
    difference, so objects that are similarly bright but differently colored
    than the background still show up in the mask.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diff_intensity = cv2.absdiff(gray, background_gray)
    diff_color = np.max(cv2.absdiff(frame, background_color), axis=2)
    diff = cv2.max(diff_intensity, diff_color)

    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    mask = cv2.medianBlur(mask, 5)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def exclude_skin_hue(hsv, mask, hue_range=(0, 20)):
    """
    Remove likely skin-hue pixels (e.g. a hand) from a foreground mask.

    Assumes skin falls in the hue range [hue_range[0], hue_range[1]].
    """
    skin_mask = cv2.inRange(hsv[:, :, 0], (hue_range[0] % 360), (hue_range[1] % 360))
    return cv2.bitwise_and(mask, cv2.bitwise_not(skin_mask))


def clean_mask(mask):
    """Remove small specks and holes from a foreground mask."""
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def find_foreground_objects(foreground_mask, min_area=1000):
    """Return contours (with statistics) found in a foreground mask cleaned by clean_mask."""
    objects = []
    mask = foreground_mask

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        moments = cv2.moments(contour)
        if moments['m00'] == 0:
            x, y, w, h = cv2.boundingRect(contour)
            centroid = (x + w // 2, y + h // 2)
        else:
            centroid = (
                int(moments['m10'] / moments['m00']),
                int(moments['m01'] / moments['m00']),
            )

        object_mask = np.zeros(foreground_mask.shape, dtype=np.uint8)
        cv2.drawContours(object_mask, [contour], -1, 255, cv2.FILLED)

        circle, circle_fit = fit_circle_to_contour(contour, object_mask)

        objects.append({
            'mask': object_mask,
            'contour': contour,
            'centroid': centroid,
            'area': area,
            'circle': circle,
            'circle_fit': circle_fit,
        })

    return objects


def mean_hue_band(hsv, mask, num_samples=20, band=10):
    """
    Return a +-band hue range around the circular mean hue inside the mask.

    Samples points inside the mask; the range is wrapped around [0, 360].
    """
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None

    num_samples = min(num_samples, len(xs))
    indices = np.random.choice(len(xs), num_samples, replace=False)
    hues = hsv[ys[indices], xs[indices], 0].astype(np.float64)

    angles = np.deg2rad(hues)
    mean_angle = np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))
    mean_hue = np.rad2deg(mean_angle) % 360

    lower_hue = (mean_hue - band) % 360
    upper_hue = (mean_hue + band) % 360
    return lower_hue, upper_hue


def fit_circle_to_contour(contour, object_mask):
    """
    Use a Hough transform to find the circle that best fits a contour.

    Returns the best-fit circle as (cx, cy, radius) and a fit score in
    [0, 1] (intersection-over-union between the circle and the contour's
    mask). Returns (None, 0.0) if no circle is found.
    """
    x, y, w, h = cv2.boundingRect(contour)
    roi = object_mask[y:y + h, x:x + w]
    roi = cv2.GaussianBlur(roi, (9, 9), 2)
    radius_estimate = max(w, h) // 2

    circles = cv2.HoughCircles(
        roi,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=max(w, h),
        param1=50,
        param2=30,
        minRadius=max(radius_estimate // 2, 1),
        maxRadius=radius_estimate + 5,
    )

    if circles is None:
        return None, 0.0

    cx, cy, r = circles[0][0]
    circle = (int(x + cx), int(y + cy), int(r))

    circle_mask = np.zeros(object_mask.shape, dtype=np.uint8)
    cv2.circle(circle_mask, circle[:2], circle[2], 255, cv2.FILLED)

    intersection = cv2.countNonZero(cv2.bitwise_and(object_mask, circle_mask))
    union = cv2.countNonZero(cv2.bitwise_or(object_mask, circle_mask))
    circle_fit = intersection / union if union > 0 else 0.0

    return circle, circle_fit


def refine_contour_by_hue_and_circle(hsv, foreground_mask, obj, hue_range):
    """
    Refine a candidate object by intersecting its hue-matched foreground with its circle.

    Returns (refined_contour, circumscribing_circle), or (None, None) if the
    object has no fitted circle, no hue range, or no overlap between the two.
    """
    if hue_range is None or obj['circle'] is None:
        return None, None

    lower_hue, upper_hue = hue_range
    hue_mask = cv2.inRange(hsv[:, :, 0], lower_hue, upper_hue)
    mask1 = cv2.bitwise_and(hue_mask, foreground_mask)

    circle_mask = np.zeros(foreground_mask.shape, dtype=np.uint8)
    cx, cy, r = obj['circle']
    cv2.circle(circle_mask, (cx, cy), r, 255, cv2.FILLED)

    mask2 = cv2.bitwise_and(mask1, circle_mask)

    contours2, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours2:
        return None, None

    refined_contour = max(contours2, key=cv2.contourArea)
    (rcx, rcy), r2 = cv2.minEnclosingCircle(refined_contour)
    circumscribing_circle = (int(rcx), int(rcy), int(r2))
    return refined_contour, circumscribing_circle


def find_hue_matches(hsv, foreground_mask, hue_range, min_area=500):
    """
    Find contours in the foreground mask that fall within a hue range.

    Unlike refine_contour_by_hue_and_circle, this does not run a Hough
    circle test - it just intersects the hue mask with the foreground mask
    and returns each resulting contour with its enclosing circle, plus the
    intersected mask (None when there is no hue range).
    """
    if hue_range is None:
        return [], None

    lower_hue, upper_hue = hue_range
    hue_mask = cv2.inRange(hsv[:, :, 0], lower_hue, upper_hue)
    mask = cv2.bitwise_and(hue_mask, foreground_mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    matches = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        (cx, cy), r = cv2.minEnclosingCircle(contour)
        matches.append((contour, (int(cx), int(cy), int(r))))

    return matches, mask


def pixel_point(center, radius):
    """Return a Point with the object center in image pixels and z as the radius in pixels."""
    return Point(x=float(center[0]), y=float(center[1]), z=float(radius))


def normalize_point(frame, point):
    """
    Normalize a pixel Point from pixel_point.

    x, y are normalized to [-1, 1] with (0, 0) at the image center, +x right
    and +y up. z (the radius in pixels) is unchanged.
    """
    height, width = frame.shape[:2]
    half_width = width / 2.0
    half_height = height / 2.0
    x = (point.x - half_width) / half_width
    y = (half_height - point.y) / half_height
    return Point(
        x=float(np.clip(x, -1.0, 1.0)),
        y=float(np.clip(y, -1.0, 1.0)),
        z=point.z)


class FindObject(Node):
    """
    Track the most circular hue-matched object in compressed camera images.

    Publishes the tracked object as a Point: x, y are the object's center
    normalized to [-1, 1] (0, 0 is the image center, +x is right, +y is up)
    and z is its radius in pixels (0 when no radius is known).

    Also publishes a debug mask for each image as a PNG CompressedImage on
    /debug_mask: the cleaned foreground mask while looking for an object, the
    pixels inside the tracked hue band while tracking, and all black while
    capturing the background.

    Also publishes an ImagePoint on /debug_viz for every received image, with
    the compressed image and the object's raw pixel center (x, y) and radius
    (z). When no object is found the point is (-1, -1, -1).
    """

    def __init__(self):
        super().__init__('find_object')
        # Show OpenCV debug windows (requires a display).
        self.display = self.declare_parameter('display', False).value

        self.image_topic = self.declare_parameter(
            'image_topic', '/image_raw/compressed').value

        # Keep background subtraction on while TRACKING too (it is always on
        # while looking for the object).
        self.static = self.declare_parameter('static', False).value

        self.state = State()

        if self.display:
            cv2.namedWindow('circle tracking', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('circle tracking', 800, 600)

        self.image_sub = self.create_subscription(
            CompressedImage,
            self.image_topic,
            self.image_callback,
            qos_profile_sensor_data)
        self.object_pub = self.create_publisher(Point, '/object', 10)
        self.debug_viz_pub = self.create_publisher(ImagePoint, '/debug_viz', 10)
        self.debug_mask_pub = self.create_publisher(CompressedImage, '/debug_mask', 10)
        self.foreground_mask = None

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
        self.debug_viz_pub.publish(
            ImagePoint(image=msg, point=NO_OBJECT_POINT if point is None else point))
        self.publish_mask(msg.header, frame.shape[:2])

        if prev_status != self.state.status:
            self.get_logger().info(f'Status: {self.state.status}')
        if self.display:
            cv2.waitKey(1)

    def publish_mask(self, header, shape):
        """Publish the foreground mask used for this frame (all black when there is none)."""
        mask = self.foreground_mask
        if mask is None:
            mask = np.zeros(shape, dtype=np.uint8)
        ok, png = cv2.imencode('.png', mask)
        if ok:
            self.debug_mask_pub.publish(
                CompressedImage(header=header, format='png', data=png.tobytes()))

    def show(self, name, image):
        if self.display:
            cv2.imshow(name, image)

    def detect_object(self, frame):
        """Advance the tracking state machine; return the object pixel Point, or None."""
        state = self.state

        if state.status == Status.BACKGROUND:
            self.capture_background(frame)
            return None

        if state.status == Status.OBJECT:
            return self.find_new_object(frame)

        if state.status == Status.TRACKING:
            return self.track_object(frame)

        return None

    def capture_background(self, frame):
        """Average a few frames into a background; runs before every OBJECT phase."""
        state = self.state
        if state.background_gray_buffer and \
                state.background_gray_buffer[0].shape != frame.shape[:2]:
            state.background_gray_buffer = []
            state.background_color_buffer = []
        state.background_gray_buffer.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        state.background_color_buffer.append(frame)
        if len(state.background_gray_buffer) < BACKGROUND_FRAMES:
            return

        state.background_gray = np.mean(
            state.background_gray_buffer, axis=0).astype(np.uint8)
        state.background_color = np.mean(
            state.background_color_buffer, axis=0).astype(np.uint8)
        state.background_gray_buffer = []
        state.background_color_buffer = []
        state.status = Status.OBJECT

    def find_new_object(self, frame):
        state = self.state
        if state.background_gray.shape != frame.shape[:2]:
            self.get_logger().warn('Image size changed, recapturing background')
            state.status = Status.BACKGROUND
            return None

        foreground_mask = background_foreground(
            frame, state.background_gray, state.background_color)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        foreground_mask = exclude_skin_hue(hsv, foreground_mask)

        cleaned_mask = clean_mask(foreground_mask)
        self.foreground_mask = cleaned_mask
        objects = find_foreground_objects(cleaned_mask)

        if self.display:
            for obj in objects:
                if obj['circle_fit'] < CIRCLE_FIT_THRESHOLD:
                    continue
                x, y, w, h = cv2.boundingRect(obj['contour'])
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{obj['circle_fit']:.2f}",
                    (x, max(0, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

        best_object = max(objects, key=lambda obj: obj['circle_fit'], default=None)
        if best_object is not None and best_object['circle_fit'] < CIRCLE_FIT_THRESHOLD:
            best_object = None

        if best_object is None:
            self.show('circle tracking', frame)
            return None

        hue_range = mean_hue_band(hsv, best_object['mask'])

        refined_contour, circumscribing_circle = refine_contour_by_hue_and_circle(
            hsv, foreground_mask, best_object, hue_range,
        )
        if refined_contour is None:
            refined_contour = best_object['contour']

        if self.display:
            contour_image = frame.copy()
            cv2.drawContours(contour_image, [refined_contour], -1, (0, 0, 255), 2)
            if circumscribing_circle is not None:
                cx, cy, r = circumscribing_circle
                cv2.circle(contour_image, (cx, cy), r, (0, 255, 0), 10)
            self.show('circle tracking', contour_image)

        state.hue_range = hue_range
        if circumscribing_circle is not None:
            state.tracking_center = circumscribing_circle[:2]
            radius = circumscribing_circle[2]
        elif best_object['circle'] is not None:
            state.tracking_center = best_object['circle'][:2]
            radius = best_object['circle'][2]
        else:
            state.tracking_center = best_object['centroid']
            radius = 0
        state.status = Status.TRACKING

        return pixel_point(state.tracking_center, radius)

    def track_object(self, frame):
        state = self.state
        if self.static and state.background_gray.shape == frame.shape[:2]:
            foreground_mask = background_foreground(
                frame, state.background_gray, state.background_color)
        else:
            foreground_mask = full_foreground(frame)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        foreground_mask = exclude_skin_hue(hsv, foreground_mask)

        matches, self.foreground_mask = find_hue_matches(
            hsv, foreground_mask, state.hue_range)

        best_match = None
        best_distance = None
        for contour, circle in matches:
            cx, cy = circle[:2]
            distance = np.hypot(
                cx - state.tracking_center[0],
                cy - state.tracking_center[1],
            )
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_match = (contour, circle)

        if best_match is None:
            self.get_logger().info('No matching hue range found')
            state.status = Status.BACKGROUND
            self.show('circle tracking', frame)
            return None

        if best_distance > DISTANCE_THRESHOLD:
            self.get_logger().info(f'Best candidate too far: {best_distance:.2f}')
            state.status = Status.BACKGROUND
            self.show('circle tracking', frame)
            return None

        contour, circle = best_match
        state.tracking_center = circle[:2]

        if self.display:
            contour_image = frame.copy()
            cv2.drawContours(contour_image, [contour], -1, (0, 0, 255), 2)
            cx, cy, r = circle
            cv2.circle(contour_image, (cx, cy), r, (0, 255, 0), 10)
            self.show('circle tracking', contour_image)

        return pixel_point(circle[:2], circle[2])


def main(args=None):
    rclpy.init(args=args)
    node = FindObject()
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
