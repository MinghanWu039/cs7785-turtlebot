# CS7785 TurtleBot object follower

The robot finds a circular, coloured object in the camera image and turns to face it.
Everything runs on the robot (ROS 2 Humble). The laptop only views the debug image,
over an ssh tunnel, so no DDS traffic crosses between the two machines.

## Build (robot)

```bash
git clone --recurse-submodules <repo-url> ~/cs7785-turtlebot
cd ~/cs7785-turtlebot
colcon build --packages-skip turtlebot3_gazebo
source install/setup.bash
```

## Run

Every robot terminal needs the workspace sourced and the same middleware:

```bash
source ~/cs7785-turtlebot/install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export TURTLEBOT3_MODEL=burger   # burger | waffle | waffle_pi
```

### Robot (over ssh), one terminal each

1. Robot bringup and camera (your existing launch file). Camera images must appear on
   `/image_raw/compressed`:
   ```bash
   ros2 launch turtlebot3_bringup camera_robot.launch.py 
   ```
2. Object detection and turning:
   ```bash
   ros2 launch loki_object_follower object_follower.launch.py
   ```
   Starts `find_object` and `rotate_robot`. To use a different camera topic:
   `ros2 run loki_object_follower find_object --ros-args -p image_topic:=<topic>`
3. Debug image server (listens on `127.0.0.1:5555`):
   ```bash
   ros2 run loki_object_follower debug_viz_server
   ```

### Laptop, two terminals

1. Tunnel to the robot's debug port (leave it running):
   ```bash
   ssh -N -L 5555:localhost:5555 <user>@<robot>
   ```
2. Viewer. Needs only `opencv-python` and `numpy`, no ROS. Press `q` to quit:
   ```bash
   python3 scripts/debug_viz_client.py
   ```
   A red "no object" label means the camera works but nothing was detected. A green
   circle marks the detected object.

### Odometry trajectory (Lab 4)

On the robot, with bringup running, start `vlio_node` and the odometry server (listens on
`127.0.0.1:5556`):
```bash
ros2 launch loki_navigate_to_goal odom_viz.launch.py
```
`vlio_node` refines wheel odometry with lidar. For each `/scan` it starts from the
odometry motion since the previous scan, runs 10 point-to-line ICP iterations against
the previous scan (ignoring point pairs more than 0.5 m apart), and publishes the result
on `/vlio_odom` at the scan rate (about 5 Hz). If too few points match, that scan uses
the odometry step instead. Run `vlio_node` with `-p icp_method:=point_to_point` for the
plain point-to-point variant. The server plots `/vlio_odom`; add
`odom_topic:=/odom` to the launch command to plot raw wheel odometry instead.

The server reports poses relative to the first message it receives, so the run starts at
(0, 0) facing +x. Run `ros2 run loki_navigate_to_goal odom_viz_server --ros-args -p
zero_start:=false` for the unshifted pose. Restarting the server resets the origin.

On the laptop, open a tunnel and start the viewer. Press `q` to quit:
```bash
ssh -N -L 5556:localhost:5556 <user>@<robot>
python3 scripts/odom_viz_client.py --waypoints wayPoints.txt   # --waypoints is optional
```
The view is a fixed 10 m x 10 m canvas centred on the start, with a 1 m grid
(`--canvas <metres>` changes it). The blue line is the path, the green dot the start, and
the red arrow the current pose. Orange circles are the waypoints, drawn with the lab's
10/15/20 cm stopping tolerances.

Grey dots are a short-lived map. Every 0.5 s the server places one `/scan` in the
plotted frame, using the pose at that scan's timestamp. It keeps the last 10 of these
frames (5 s); older frames are drawn lighter. Map points outside the 10 m canvas are
dropped. Server parameters: `map_period` (0.5), `map_frames` (10), `map_size` (10.0),
`scan_topic` (`/scan`).

## Checks

```bash
ros2 topic hz /image_raw/compressed   # camera is publishing
ros2 topic hz /debug_viz              # find_object is publishing (one message per frame)
ros2 topic echo /object               # only while an object is detected
```

## Topics

| Topic | Type | Published by |
|---|---|---|
| `/image_raw/compressed` | `sensor_msgs/CompressedImage` | camera |
| `/object` | `geometry_msgs/Point` (x, y in [-1, 1], z = radius px) | `find_object` |
| `/debug_viz` | `loki_object_follower_msgs/ImagePoint` (radius < 0 = no object) | `find_object` |
| `/debug_mask` | `sensor_msgs/CompressedImage` (PNG foreground mask) | `find_object` |
| `/cmd_vel` | `geometry_msgs/Twist` | `rotate_robot` |
| `/scan` | `sensor_msgs/LaserScan` (360 beams, 0 = no return) | LDS lidar |
| `/odom` | `nav_msgs/Odometry` | TurtleBot3 bringup |
| `/vlio_odom` | `nav_msgs/Odometry` (ICP-refined, ~5 Hz) | `vlio_node` |

## Notes

- Do not run `ros2` commands on a laptop with a different ROS distro while the robot
  nodes run. Mixed Humble/Jazzy participants crashed robot nodes.
- `rotate_robot` publishes `Twist` on `/cmd_vel`. Check that your robot driver subscribes to
  that type (`ros2 topic info -v /cmd_vel`); Jazzy TurtleBot3 drivers expect `TwistStamped`.
- `debug_viz` (`ros2 run loki_object_follower debug_viz`) is a ROS-based viewer for a
  machine on the same DDS network as the robot, as an alternative to the ssh tunnel.
