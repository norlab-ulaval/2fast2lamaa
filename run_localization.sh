#!/bin/bash
# Run 2Fast-2Lamaa in pure localization mode against the map in /mapping,
# replaying the rosbag in /data, and save + plot the resulting trajectory.
#
# Usage: ./run_localization.sh [BAG_PATH]
#   BAG_PATH defaults to /data.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MAPPING_DIR="/mapping"
MAPPING_BACKUP="/mapping_bcap"
BAG_PATH="${1:-/data}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/output}"
PLOT_SCRIPT="${SCRIPT_DIR}/scripts/plot_trajectory.py"
# Seconds of bag time to replay; empty or 0 replays the whole bag.
PLAYBACK_DURATION="${PLAYBACK_DURATION:-600}"

export NAMESPACE="${NAMESPACE:-}"
export IMU_TYPE="${IMU_TYPE:-vectornav}"
export STORAGE_PATH="${MAPPING_DIR}"

NODE_STARTUP_TIMEOUT=60
SHUTDOWN_TIMEOUT=90

LAUNCH_PID=""

cleanup() {
    if [ -n "${LAUNCH_PID}" ] && kill -0 "${LAUNCH_PID}" 2>/dev/null; then
        kill -INT "${LAUNCH_PID}" 2>/dev/null || true
        sleep 2
        kill -9 "${LAUNCH_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

if [ ! -d "${MAPPING_BACKUP}" ]; then
    echo "Backup mapping directory ${MAPPING_BACKUP} not found." >&2
    exit 1
fi
if [ ! -e "${BAG_PATH}" ]; then
    echo "Rosbag path ${BAG_PATH} not found." >&2
    exit 1
fi

echo "Restoring ${MAPPING_DIR} from ${MAPPING_BACKUP}..."
rm -rf "${MAPPING_DIR:?}"/*
cp -a "${MAPPING_BACKUP}/." "${MAPPING_DIR}/"

mkdir -p "${OUTPUT_DIR}"

set +u
# shellcheck disable=SC1091
source "/opt/ros/${ROS_DISTRO}/setup.bash"
# shellcheck disable=SC1091
source "/ros2_ws/install/setup.bash"
set -u

echo "Launching 2Fast-2Lamaa localization (map: ${STORAGE_PATH})..."
ros2 launch ros_launchers 2fast2lamaa_localization.launch.py use_sim_time:=true rviz:=false &
LAUNCH_PID=$!

echo "Waiting for localization nodes to come up..."
waited=0
while ! ros2 node list 2>/dev/null | grep -q "/${NAMESPACE:+$NAMESPACE/}gp_map"; do
    sleep 1
    waited=$((waited + 1))
    if [ "${waited}" -ge "${NODE_STARTUP_TIMEOUT}" ]; then
        echo "Timed out waiting for gp_map node to start." >&2
        exit 1
    fi
    if ! kill -0 "${LAUNCH_PID}" 2>/dev/null; then
        echo "Launch process exited unexpectedly." >&2
        exit 1
    fi
done
echo "Localization nodes are up."

play_args=("${BAG_PATH}" --clock)
if [ -n "${PLAYBACK_DURATION}" ] && [ "${PLAYBACK_DURATION}" != "0" ]; then
    play_args+=(--playback-duration "${PLAYBACK_DURATION}")
    echo "Replaying bag ${BAG_PATH} (first ${PLAYBACK_DURATION} s)..."
else
    echo "Replaying bag ${BAG_PATH} (full)..."
fi
ros2 bag play "${play_args[@]}"

echo "Bag playback finished; shutting down localization nodes..."
kill -INT "${LAUNCH_PID}"
waited=0
while kill -0 "${LAUNCH_PID}" 2>/dev/null; do
    sleep 1
    waited=$((waited + 1))
    if [ "${waited}" -ge "${SHUTDOWN_TIMEOUT}" ]; then
        echo "Localization nodes did not shut down in time; forcing kill." >&2
        kill -9 "${LAUNCH_PID}" 2>/dev/null || true
        break
    fi
done
wait "${LAUNCH_PID}" 2>/dev/null || true
LAUNCH_PID=""

echo "Collecting trajectory output into ${OUTPUT_DIR}..."
shopt -s nullglob
traj_files=("${MAPPING_DIR}"/trajectory_submap_*.csv "${MAPPING_DIR}/trajectory_map.csv" "${MAPPING_DIR}/trajectory.txt")
shopt -u nullglob
if [ "${#traj_files[@]}" -eq 0 ]; then
    echo "No trajectory output files found in ${MAPPING_DIR}." >&2
    exit 1
fi
cp -a "${traj_files[@]}" "${OUTPUT_DIR}/"

echo "Plotting trajectory..."
python3 "${PLOT_SCRIPT}" "${OUTPUT_DIR}" -o "${OUTPUT_DIR}/trajectory.png"

echo "Done. Output in ${OUTPUT_DIR}"
