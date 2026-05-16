import json
import argparse

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped


class GraspPosePublisher(Node):
    def __init__(self, command_json):
        super().__init__("grasp_pose_publisher")
        self.pub = self.create_publisher(PoseStamped, "/lg_risk_mfenet/grasp_pose", 10)

        with open(command_json, "r", encoding="utf-8") as f:
            self.cmd = json.load(f)

        self.timer = self.create_timer(1.0, self.publish_pose)

    def publish_pose(self):
        grasp = self.cmd["grasp_pose"]
        pos = grasp["position"]
        quat = grasp["orientation"]

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.cmd.get("frame_id", "camera_link")

        msg.pose.position.x = float(pos["x"])
        msg.pose.position.y = float(pos["y"])
        msg.pose.position.z = float(pos["z"])

        msg.pose.orientation.x = float(quat["x"])
        msg.pose.orientation.y = float(quat["y"])
        msg.pose.orientation.z = float(quat["z"])
        msg.pose.orientation.w = float(quat["w"])

        self.pub.publish(msg)
        self.get_logger().info(
            f"Published grasp pose: frame={msg.header.frame_id}, "
            f"pos=({msg.pose.position.x:.3f}, {msg.pose.position.y:.3f}, {msg.pose.position.z:.3f}), "
            f"quat=({msg.pose.orientation.x:.3f}, {msg.pose.orientation.y:.3f}, "
            f"{msg.pose.orientation.z:.3f}, {msg.pose.orientation.w:.3f})"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--command-json", type=str, required=True)
    args = parser.parse_args()

    rclpy.init()
    node = GraspPosePublisher(args.command_json)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
