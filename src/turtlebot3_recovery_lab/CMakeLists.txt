cmake_minimum_required(VERSION 3.8)
project(turtlebot3_recovery_lab)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -std=c++17)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav2_behavior_tree REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(turtlebot3_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Generate custom service
rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/DivertRobot.srv"
  DEPENDENCIES geometry_msgs
)

# Build C++ BT Plugin Library
add_library(beep_and_divert_bridge_plugin SHARED
  src/beep_and_divert_bridge.cpp
)

target_compile_definitions(
  beep_and_divert_bridge_plugin
  PRIVATE BT_PLUGIN_EXPORT
)

target_include_directories(beep_and_divert_bridge_plugin PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(beep_and_divert_bridge_plugin
  rclcpp
  nav2_behavior_tree
  geometry_msgs
  turtlebot3_msgs
)

rosidl_get_typesupport_target(
  cpp_typesupport_target
  "${PROJECT_NAME}"
  "rosidl_typesupport_cpp"
)

target_link_libraries(
  beep_and_divert_bridge_plugin
  "${cpp_typesupport_target}"
)

# Install Python package
ament_python_install_package(turtlebot3_recovery_nodes)

# Install targets
install(TARGETS beep_and_divert_bridge_plugin
  DESTINATION lib
)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

install(PROGRAMS
  turtlebot3_recovery_nodes/recovery.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install behavior trees
install(DIRECTORY behavior_trees/
  DESTINATION share/${PROJECT_NAME}/behavior_trees
)

# Export dependencies and libraries
ament_export_include_directories(include)
ament_export_libraries(beep_and_divert_bridge_plugin)
ament_export_dependencies(
  rclcpp
  nav2_behavior_tree
  geometry_msgs
  turtlebot3_msgs
  rosidl_default_runtime
)

ament_package()
