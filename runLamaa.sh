#!/bin/bash

docker run -it \
  --name slam \
  -it \
  --rm \
  --privileged \
  --network host \
  -e SSH_AUTH_SOCK=${SSH_AUTH_SOCK} \
  -v ../../ros_launchers:/ros2_ws/src/ros_launchers \
  -v /home/mabox/data/2024-11-21/red_2024-11-21-10-34:/data \
  -v /home/mabox/2fast2lamaa:/ros2_ws/src/2fast2lamaa \
  -e NAMESPACE="" \
  -e IS_MAPPING=1 \
  -e STORAGE_PATH=/ros2_ws/src/2fast2lamaa/output \
  norlab/2fast2lamaa /bin/bash # -c "source /ros2_ws/src/2fast2lamaa/install/setup.bash && ros2 launch ffastllamaa fomo.launch.py bag_path:=/data"


  # -v ../../ros_launchers:/root/ASRL/vtr3/src/main/src/ros_launchers \
  # -v ../../ros_launchers:/ros2_ws/src/ros_launchers \
  # -v /home/mbo/bigfoot-FoMo/mcap:/rosbags \