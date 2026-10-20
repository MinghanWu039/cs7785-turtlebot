from enum import Enum
import numpy as np

class State(Enum):
    GO_TO_GOAL = 1
    AVOID_OBSTACLE = 2
    FOLLOW_BOUNDARY_C = 3
    FOLLOW_BOUNDARY_CC = 4

class StateMachine:
    def __init__(self, initialState):
        self.cur_state = initialState
        self.epsilon = 0.05
        self.safe_dist = 0.2
        self.last_goal_dist = 0.0

    def update_state(self, u_state: dict[State, np.ndarray], obs_dist, goal_dist):
        if self.cur_state == State.GO_TO_GOAL:
            if obs_dist <= self.safe_dist:
                if np.dot(u_state[State.FOLLOW_BOUNDARY_C], u_state[State.GO_TO_GOAL]) > 0:
                    self.cur_state = State.FOLLOW_BOUNDARY_C
                    self.last_goal_dist = goal_dist
                elif np.dot(u_state[State.FOLLOW_BOUNDARY_CC], u_state[State.GO_TO_GOAL]) > 0:
                    self.cur_state = State.FOLLOW_BOUNDARY_CC
                    self.last_goal_dist = goal_dist
        if self.cur_state == State.FOLLOW_BOUNDARY_C or self.cur_state == State.FOLLOW_BOUNDARY_CC:
            if obs_dist < self.safe_dist - self.epsilon:
                self.cur_state = State.AVOID_OBSTACLE
            elif (np.dot(u_state[State.GO_TO_GOAL], u_state[State.AVOID_OBSTACLE]) > 0 and goal_dist < self.last_goal_dist) or \
                obs_dist > self.safe_dist + self.epsilon:
                self.cur_state = State.GO_TO_GOAL
        if self.cur_state == State.AVOID_OBSTACLE:
            if obs_dist > self.safe_dist:
                if np.dot(u_state[State.GO_TO_GOAL], u_state[State.FOLLOW_BOUNDARY_C]) > 0:
                    self.cur_state = State.FOLLOW_BOUNDARY_C
                    self.last_goal_dist = goal_dist
                elif np.dot(u_state[State.GO_TO_GOAL], u_state[State.FOLLOW_BOUNDARY_CC]) > 0:
                    self.cur_state = State.FOLLOW_BOUNDARY_CC
                    self.last_goal_dist = goal_dist
        return self.cur_state

def compute_control_data(pos, point_cloud, goal):
    """
    This function calculates the control vectors for each state based on the point cloud and the goal.
    """
    u_state = {}
    goal_vec = goal - pos
    g_dist = np.linalg.norm(goal_vec)
    u_state[State.GO_TO_GOAL] = goal_vec / g_dist if g_dist > 0 else np.zeros_like(goal_vec)
    obs_vec = get_obstacle_vector(pos, point_cloud)
    if obs_vec is None:  # nothing in range: no obstacle to avoid
        obs_vec, obs_dist = np.zeros_like(goal_vec), np.inf
    else:
        obs_dist = np.linalg.norm(obs_vec)
    u_state[State.AVOID_OBSTACLE] = -obs_vec / obs_dist if 0 < obs_dist < np.inf else np.zeros_like(obs_vec)
    R_90 = np.array([[0, -1], [1, 0]])
    u_state[State.FOLLOW_BOUNDARY_C] = R_90 @ u_state[State.AVOID_OBSTACLE]
    u_state[State.FOLLOW_BOUNDARY_CC] = -R_90 @ u_state[State.AVOID_OBSTACLE]
    return u_state, obs_dist, g_dist

def get_obstacle_vector(pos, point_cloud, max_gap=0.3):
    """
    Assuming the point cloud is an ordered 2D array of shape (N, 2) where N is the number of points and each
    point has x and y coordinates, this function returns a vector that points towards the closest obstacle point.
    Consider the point cloud as polyline segments.
    Segments longer than max_gap are skipped: they bridge missing returns rather than a real surface, and a
    wide gap would otherwise put a phantom wall right next to the robot. Every point also counts on its own,
    so isolated returns are still obstacles. Returns None for an empty point cloud.
    """
    if len(point_cloud) == 0:
        return None
    def closest_point_on_segment(p, a, b):
        """Returns the closest point on the line segment ab to point p."""
        ab = b - a
        t = np.dot(p - a, ab) / np.dot(ab, ab)
        t = np.clip(t, 0, 1)  # Clamp t to the range [0, 1]
        return a + t * ab
    candidates = [point_cloud]
    polyline = np.stack((point_cloud[:-1], point_cloud[1:]), axis=1)
    lengths = np.linalg.norm(polyline[:, 1] - polyline[:, 0], axis=1)
    polyline = polyline[(lengths > 0) & (lengths <= max_gap)]
    if len(polyline):
        candidates.append(np.vectorize(closest_point_on_segment, signature='(n),(n),(n)->(n)')(pos, polyline[:, 0], polyline[:, 1]))
    closest_points = np.vstack(candidates)
    closest_point = closest_points[np.argmin(np.linalg.norm(closest_points - pos, axis=1))]
    obstacle_vector = closest_point - pos
    return obstacle_vector
    
