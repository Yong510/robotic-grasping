import json
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--command-json", required=True)
    args = parser.parse_args()

    with open(args.command_json, "r", encoding="utf-8") as f:
        cmd = json.load(f)

    print("========== Mock Robot Execution ==========")
    print(f"Task: {cmd.get('task')}")
    print(f"Frame: {cmd.get('frame_id')}")
    print(f"Object: id={cmd.get('selected_object_id')}, class={cmd.get('selected_class')}")
    print(f"Target place: {cmd.get('target_place')}")

    pre = cmd["pregrasp_pose"]["position"]
    grasp = cmd["grasp_pose"]["position"]
    quat = cmd["grasp_pose"]["orientation"]
    gripper = cmd["gripper"]

    print("\n[1] Move to pre-grasp pose")
    print(f"    x={pre['x']:.4f}, y={pre['y']:.4f}, z={pre['z']:.4f}")

    print("\n[2] Move down to grasp pose")
    print(f"    x={grasp['x']:.4f}, y={grasp['y']:.4f}, z={grasp['z']:.4f}")

    print("\n[3] Align gripper orientation")
    print(f"    q=({quat['x']:.4f}, {quat['y']:.4f}, {quat['z']:.4f}, {quat['w']:.4f})")

    print("\n[4] Close gripper")
    print(f"    width={gripper['width']:.4f}")

    print("\n[5] Lift object")
    print(f"    lift z={pre['z']:.4f}")

    print("\n[6] Move to target place")
    print(f"    target={cmd.get('target_place')}")

    print("\n[OK] Mock execution command is valid.")


if __name__ == "__main__":
    main()
