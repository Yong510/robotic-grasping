#!/bin/bash
set -e

# ROS2 Humble + MoveIt2 setup script for Ubuntu 22.04.
# Run as root in a fresh AutoDL Ubuntu 22.04 instance.

if [ ! -f /etc/os-release ]; then
  echo "Cannot find /etc/os-release"
  exit 1
fi

. /etc/os-release

echo "[INFO] OS: $PRETTY_NAME"

if [ "$VERSION_ID" != "22.04" ]; then
  echo "[WARN] This script is designed for Ubuntu 22.04. Current VERSION_ID=$VERSION_ID"
  echo "[WARN] ROS2 Humble is recommended on Ubuntu 22.04. Continue only if you know what you are doing."
fi

apt-get update
apt-get install -y software-properties-common curl gnupg lsb-release locales

locale-gen en_US en_US.UTF-8
update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

add-apt-repository universe -y

curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
  > /etc/apt/sources.list.d/ros2.list

apt-get update
apt-get install -y \
  ros-humble-desktop \
  ros-humble-moveit \
  ros-humble-tf2-ros \
  ros-humble-tf2-geometry-msgs \
  ros-humble-robot-state-publisher \
  ros-humble-joint-state-publisher-gui \
  python3-colcon-common-extensions \
  python3-rosdep \
  python3-pip \
  unzip \
  git

if ! rosdep --help >/dev/null 2>&1; then
  echo "[WARN] rosdep command not available."
else
  rosdep init 2>/dev/null || true
  rosdep update || true
fi

# Add ROS source to bashrc if not already present.
if ! grep -q "source /opt/ros/humble/setup.bash" /root/.bashrc; then
  echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc
fi

source /opt/ros/humble/setup.bash

python3 - <<'PY'
import rclpy
print('rclpy ok')
PY

echo "[DONE] ROS2 Humble + MoveIt2 installation finished."
echo "Run: source /opt/ros/humble/setup.bash"
