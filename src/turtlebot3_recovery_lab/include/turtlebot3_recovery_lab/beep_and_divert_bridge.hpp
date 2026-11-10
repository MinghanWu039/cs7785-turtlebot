#pragma once

#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav2_behavior_tree/bt_service_node.hpp"
#include "turtlebot3_recovery_lab/srv/divert_robot.hpp"

namespace turtlebot3_recovery_lab
{

class BeepAndDivertBridge
  : public nav2_behavior_tree::BtServiceNode<
      turtlebot3_recovery_lab::srv::DivertRobot>
{
public:
  using ServiceType = turtlebot3_recovery_lab::srv::DivertRobot;

  BeepAndDivertBridge(
    const std::string & name,
    const BT::NodeConfiguration & config);

  static BT::PortsList providedPorts()
  {
    return providedBasicPorts({
      BT::OutputPort<geometry_msgs::msg::PoseStamped>(
        "recovery_pose",
        "Diversion waypoint returned by the recovery service")
    });
  }

  void on_tick() override;

  BT::NodeStatus on_completion(
    std::shared_ptr<ServiceType::Response> response) override;
};

}  // namespace turtlebot3_recovery_lab