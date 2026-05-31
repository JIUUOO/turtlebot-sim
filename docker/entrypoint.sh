#!/usr/bin/env bash
set -e

source "/opt/ros/${ROS_DISTRO:-humble}/setup.bash"

if [ -f /workspace/turtlebot-sim/install/setup.bash ]; then
  source /workspace/turtlebot-sim/install/setup.bash
fi

exec "$@"
