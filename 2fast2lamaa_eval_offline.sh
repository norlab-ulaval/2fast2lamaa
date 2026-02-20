#!/bin/bash
trajectory="green"

input_path_host=/home/mabox/ssd
output_path_host="/home/mabox/output/2fast2lamaa-${trajectory}"

for date_dir in "${input_path_host}"/*/; do
    date=$(basename "${date_dir}")
    for dataset_dir in "${date_dir}${trajectory}_"*/; do
        [ -d "${dataset_dir}" ] || continue
        dataset=$(basename "${dataset_dir}")
        echo $dataset
        mkdir -p "${output_path_host}/${date}/${dataset}"
        echo "Processing dataset ${date}/${dataset}"
        docker run -it --rm \
            --name slam \
            -e DATASET_PATH=/data \
            -e NAMESPACE="" \
            -e IS_MAPPING=1 \
            -e STORAGE_PATH=/output \
            -v /home/mabox/2fast2lamaa:/ros2_ws/src/2fast2lamaa \
            -v "${dataset_dir}":/data \
            -v "${output_path_host}/${date}/${dataset}":/output \
            norlab/2fast2lamaa /bin/bash -c "source /ros2_ws/src/2fast2lamaa/install/setup.bash && ros2 launch ffastllamaa fomo.launch.py bag_path:=/data"
        docker run -it --rm \
            --name slam \
            -e DATASET_PATH=/data \
            -e NAMESPACE="" \
            -e IS_MAPPING=1 \
            -e STORAGE_PATH=/output \
            -v /home/mabox/2fast2lamaa:/ros2_ws/src/2fast2lamaa \
            -v "${dataset_dir}":/data \
            -v "${output_path_host}/${date}/${dataset}":/output \
            norlab/2fast2lamaa /bin/bash -c "source /ros2_ws/src/2fast2lamaa/install/setup.bash && python3 /ros2_ws/src/2fast2lamaa/scripts/loop_closure.py /output"
    done
done

# for date_dir in "${input_path_host}"/*/; do
#     date=$(basename "${date_dir}")
#     # Skip folders before June 2025
#     if [[ "${date}" < "2025-06" ]]; then
#         continue
#     fi
#     for dataset_dir in "${date_dir}${trajectory}_"*/; do
#         [ -d "${dataset_dir}" ] || continue
#         dataset=$(basename "${dataset_dir}")
#         mkdir -p "${output_path_host}/${date}/${dataset}"
#         echo "Processing dataset ${date}/${dataset}"
#         docker run -it --rm \
#             -v "${dataset_dir}":/dataset \
#             -v "${output_path_host}/${date}/${dataset}":/output \
#             -e DATASET_PATH=/dataset \
#             -e OUTPUT_PATH=/output \
#             norlab/2fast2lamaa run_orbslam3 \
#             2>&1 | tee "${output_path_host}/${date}/${dataset}/docker.log"
#     done
# done


# input_path_host=/home/mabox/data
# output_path_host="/home/mabox/output/2fast2lamaa-${trajectory}"
# date="2024-11-21"
# dataset="red_2024-11-21-10-34"
# dataset_dir="${input_path_host}/${date}/${dataset}"

# docker run -it --rm \
#     --name slam \
#     -e DATASET_PATH=/data \
#     -e NAMESPACE="" \
#     -e IS_MAPPING=1 \
#     -e STORAGE_PATH=/output \
#     -v /home/mabox/2fast2lamaa:/ros2_ws/src/2fast2lamaa \
#     -v "${dataset_dir}":/data \
#     -v "${output_path_host}/${date}/${dataset}":/output \
#     norlab/2fast2lamaa /bin/bash -c "source /opt/ros/humble/setup.bash && source /ros2_ws/src/2fast2lamaa/install/setup.bash && ros2 launch ffastllamaa fomo.launch.py bag_path:=/data"
# docker run -it --rm \
#     --name slam \
#     -e DATASET_PATH=/data \
#     -e NAMESPACE="" \
#     -e IS_MAPPING=1 \
#     -e STORAGE_PATH=/output \
#     -v /home/mabox/2fast2lamaa:/ros2_ws/src/2fast2lamaa \
#     -v "${dataset_dir}":/data \
#     -v "${output_path_host}/${date}/${dataset}":/output \
#     norlab/2fast2lamaa /bin/bash -c "source /opt/ros/humble/setup.bash && source /ros2_ws/src/2fast2lamaa/install/setup.bash && python3 /ros2_ws/src/2fast2lamaa/scripts/loop_closure.py /output"