#include "rclcpp/rclcpp.hpp"
#include "rclcpp/serialization.hpp"
#include "ros_utils.h"
#include "lice/utils.h"
#include "lice/types.h"
#include "lice/lidar_odometry.h"
#include "lice/math_utils.h"
#include "lice/pointcloud_utils.h"
#include "lice/submap_manager.h"

#include <memory>
#include <thread>
#include <atomic>
#include <mutex>
#include <sys/stat.h>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <map>

#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/msg/imu.hpp"

#include "rosbag2_cpp/reader.hpp"


bool folderExistsOB(const std::string& folderPath) {
    struct stat info;
    if (stat(folderPath.c_str(), &info) != 0) return false;
    else if (info.st_mode & S_IFDIR) return true;
    else return false;
}

bool createFolderOB(const std::string& folderPath) {
    mode_t mode = 0755;
    int ret = mkdir(folderPath.c_str(), mode);
    if (ret == 0) return true;
    return false;
}


class OfflineBoreasProcessorNode : public rclcpp::Node, public LidarOdometryPublisher
{
public:
    OfflineBoreasProcessorNode()
        : rclcpp::Node("offline_boreas_processor", rclcpp::NodeOptions().allow_undeclared_parameters(true))
    {
        RCLCPP_INFO(this->get_logger(), "Starting offline_boreas_processor node");

        std::string bag_path = readRequiredFieldString(this, "bag_path");
        lidar_topic_ = readFieldString(this, "lidar_topic", "/velodyne_points");
        imu_acc_topic_ = readFieldString(this, "imu_acc_topic", "/imu/data");
        imu_gyr_topic_ = readFieldString(this, "imu_gyr_topic", "/imu/data");

        // ======= Lidar Odometry Params =======
        LidarOdometryParams lo_params;
        lo_params.low_latency = readFieldBool(this, "low_latency", true);
        lo_params.dense_pc_output = readFieldBool(this, "dense_pc_output", false);
        lo_params.min_range = readFieldDouble(this, "min_range", 1.0);
        lo_params.max_range = readFieldDouble(this, "max_range", 150.0);
        lo_params.min_feature_dist = readFieldDouble(this, "min_feature_dist", 0.05);
        lo_params.max_feature_dist = readFieldDouble(this, "max_feature_dist", 0.5);
        lo_params.max_feature_range = readFieldDouble(this, "max_feature_range", 150.0);
        lo_params.feature_voxel_size = readFieldDouble(this, "feature_voxel_size", 0.3);
        lo_params.loss_function_scale = readFieldDouble(this, "lo_loss_function_scale", 1.0);
        lo_params.state_frequency = readFieldDouble(this, "state_freq", 200.0);
        lo_params.gyr_std = readFieldDouble(this, "gyr_std", 0.005);
        lo_params.acc_std = readFieldDouble(this, "acc_std", 0.02);
        lo_params.lidar_std = readFieldDouble(this, "lidar_std", 0.02);
        lo_params.g = readFieldDouble(this, "g", 9.80);
        
        std::string mode = readFieldString(this, "mode", "imu");
        if(kLidarOdometryModeMap.find(mode) != kLidarOdometryModeMap.end()) {
            lo_params.mode = kLidarOdometryModeMap.at(mode);
            mode_ = lo_params.mode;
        } else {
            RCLCPP_ERROR(this->get_logger(), "Invalid mode parameter: %s", mode.c_str());
            throw std::runtime_error("Invalid mode parameter");
        }

        lo_params.calib_px = readRequiredFieldDouble(this, "calib_px");
        lo_params.calib_py = readRequiredFieldDouble(this, "calib_py");
        lo_params.calib_pz = readRequiredFieldDouble(this, "calib_pz");
        lo_params.calib_rx = readRequiredFieldDouble(this, "calib_rx");
        lo_params.calib_ry = readRequiredFieldDouble(this, "calib_ry");
        lo_params.calib_rz = readRequiredFieldDouble(this, "calib_rz");

        lo_params.max_associations_per_type = readFieldInt(this, "max_associations_per_type", 1000);
        lo_params.unsorted_pc = readFieldBool(this, "unsorted_pc", false);
        lo_params.planar_only = readFieldBool(this, "planar_only", false);

        pc_scale_ = readFieldDouble(this, "point_cloud_scale", 1.0);

        std::string broken_channels_str = readFieldString(this, "broken_channels", "");
        if(!broken_channels_str.empty()) {
            std::stringstream ss(broken_channels_str);
            std::string token;
            while(std::getline(ss, token, ',')) {
                 broken_channels_.insert(std::stoi(token));
            }
        }

        acc_in_m_s2_ = readFieldBool(this, "acc_in_m_per_s2", true);
        invert_imu_ = readFieldBool(this, "invert_imu", false);
        time_field_multiplier_ = readFieldDouble(this, "point_time_multiplier", 1e-9);
        absolute_time_ = readFieldBool(this, "absolute_time", false);

        lidar_odometry_ = std::make_shared<LidarOdometry>(lo_params, this);

        // ======= GP Map Params =======
        voxel_size_ = readRequiredFieldDouble(this, "voxel_size");
        MapDistFieldOptions options;
        options.cell_size = voxel_size_;
        downsample_size_ = readFieldDouble(this, "voxel_size_factor_for_registration", 2.0) * voxel_size_;
        options.neighborhood_size = readRequiredFieldInt(this, "neighbourhood_size");

        register_ = readFieldBool(this, "register", true);
        with_init_guess_ = readRequiredFieldBool(this, "with_init_guess");
        approximate_ = readFieldBool(this, "register_with_approximate_field", false);
        options.edge_field = readFieldBool(this, "use_edge_field", true);
        use_edge_field_ = options.edge_field;

        options.use_temporal_weights = readFieldBool(this, "use_temporal_weights", false);
        options.free_space_carving_radius = readFieldDouble(this, "free_space_carving_radius", -1.0);
        options.free_space_carving = (options.free_space_carving_radius > 0.0);
        this->get_parameter("min_range", options.min_range);
        this->get_parameter("max_range", options.max_range);

        key_framing_ = readFieldBool(this, "key_framing", false);
        key_framing_dist_thr_ = readFieldDouble(this, "key_framing_dist_thr", 10.0);
        key_framing_rot_thr_ = readFieldDouble(this, "key_framing_rot_thr", 0.26);
        key_framing_time_thr_ = readFieldDouble(this, "key_framing_time_thr", 0.5);
        
        localization_ = readFieldBool(this, "localization_only", false);
        max_nb_pts_ = readFieldInt(this, "max_num_pts_for_registration", 8000);

        std::string map_path = readRequiredFieldString(this, "map_path");
        bool reverse_path = false;
        bool using_submaps = readFieldBool(this, "using_submaps", false);

        if(readFieldBool(this, "write_scans", false)) {
            options.scan_folder = map_path;
            if(options.scan_folder.back() != '/') options.scan_folder += "/";
            options.scan_folder += "scans/";
            if(folderExistsOB(options.scan_folder)) std::filesystem::remove_all(options.scan_folder);
            if(!createFolderOB(options.scan_folder)) throw std::runtime_error("Could not create scan folder");
        }

        if(localization_) {
            if(using_submaps) reverse_path = readRequiredFieldBool(this, "reverse_path");
            double init_pose_x = readFieldDouble(this, "init_pose_x", 0.0);
            double init_pose_y = readFieldDouble(this, "init_pose_y", 0.0);
            double init_pose_z = readFieldDouble(this, "init_pose_z", 0.0);
            double init_pose_rx = readFieldDouble(this, "init_pose_rx", 0.0);
            double init_pose_ry = readFieldDouble(this, "init_pose_ry", 0.0);
            double init_pose_rz = readFieldDouble(this, "init_pose_rz", 0.0);

            init_guess_ = Mat4::Identity();
            init_guess_.block<3,1>(0,3) = Vec3(init_pose_x, init_pose_y, init_pose_z);
            init_guess_.block<3,3>(0,0) = expMap(Vec3(init_pose_rx, init_pose_ry, init_pose_rz));
        }

        if(!folderExistsOB(map_path)) {
            if(!createFolderOB(map_path)) throw std::runtime_error("Could not create map folder");
        }

        options.over_reject = readFieldBool(this, "over_reject", false);
        options.last_scan_carving = readFieldBool(this, "last_scan_carving", true);

        pc_type_internal_ = readFieldBool(this, "point_cloud_internal_type", true); 
        map_loss_scale_ = readFieldDouble(this, "map_loss_function_scale", 0.5);

        traj_path_ = map_path;
        if(traj_path_.back() != '/') traj_path_ += "/";
        traj_path_ += "trajectory.txt";
        createTrajectoryFile(traj_path_);

        double submap_length = readFieldDouble(this, "submap_length", 200.0);
        double submap_overlap = readFieldDouble(this, "submap_overlap", 0.2);
        if(!localization_) {
            using_submaps = (submap_length > 0.0);
        }
        
        map_ = std::make_shared<SubmapManager>(options, localization_, using_submaps, submap_length, submap_overlap, map_path, reverse_path);

        // Start LO optimization thread
        lo_thread_ = lidar_odometry_->runThread();
        
        // Start bag parsing thread
        bag_reader_thread_ = std::make_unique<std::thread>(&OfflineBoreasProcessorNode::processBag, this, bag_path);
    }

    ~OfflineBoreasProcessorNode()
    {
        if (running_) shutdown();
    }

    void shutdown()
    {
        if (!running_.exchange(false)) return;

        if (bag_reader_thread_ && bag_reader_thread_->joinable()) {
            if (std::this_thread::get_id() != bag_reader_thread_->get_id()) {
                bag_reader_thread_->join();
            } else {
                bag_reader_thread_->detach();
            }
        }
        
        RCLCPP_INFO(this->get_logger(), "Waiting for remaining point clouds to be processed: %d pushed, %d processed", 
                    pushed_pc_count_.load(), processed_pc_count_.load());
        while(processed_pc_count_.load() < pushed_pc_count_.load() && rclcpp::ok()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }



        if (lidar_odometry_) lidar_odometry_->stop();
        if (lo_thread_ && lo_thread_->joinable()) lo_thread_->join();

        if (map_) {
            std::lock_guard<std::mutex> lock(map_mutex_);
            RCLCPP_INFO(this->get_logger(), "Writing final map to disk...");
            map_->writeMap();
        }
        
        RCLCPP_INFO(this->get_logger(), "Offline processing complete.");
    }

    // ========== LidarOdometryPublisher Overrides ==========
    void publishTransform(const int64_t /*t*/, const Vec3& /*pos*/, const Vec3& /*rot*/) override {}

    void publishGlobalOdom(const int64_t t, const Vec3& pos, const Vec3& rot, const Vec3& /*vel*/, const Vec3& /*ang_vel*/) override {
        Eigen::AngleAxisd aa(rot.norm(), rot.normalized());
        Eigen::Quaterniond q(aa);
        Mat4 pose = Mat4::Identity();
        pose.block<3,1>(0,3) = pos;
        pose.block<3,3>(0,0) = q.toRotationMatrix();
        
        std::lock_guard<std::mutex> lock(pose_mutex_);
        latest_global_odom_pose_[t] = pose;
    }

    void publishPcDense(const int64_t /*t*/, const std::vector<Pointd>& /*pc*/) override {}

    void publishPc(const int64_t t, const std::vector<Pointd>& pc) override {
        rclcpp::Time new_time(t);

        Mat4 global_pose;
        {
            std::lock_guard<std::mutex> lock(pose_mutex_);
            auto it = latest_global_odom_pose_.find(t);
            if (it != latest_global_odom_pose_.end()) {
                global_pose = it->second;
                latest_global_odom_pose_.erase(latest_global_odom_pose_.begin(), std::next(it)); // cleanup older standardly
            } else {
                global_pose = Mat4::Identity();
                RCLCPP_WARN(this->get_logger(), "No global pose found for t=%ld", t);
            }
        }

        updateMap(pc, new_time, global_pose);
        processed_pc_count_++;
    }

private:
    std::string lidar_topic_;
    std::string imu_acc_topic_;
    std::string imu_gyr_topic_;

    std::atomic<bool> running_{true};
    std::unique_ptr<std::thread> bag_reader_thread_;
    std::shared_ptr<std::thread> lo_thread_;

    std::shared_ptr<LidarOdometry> lidar_odometry_;
    std::shared_ptr<SubmapManager> map_;
    std::mutex map_mutex_;

    // LO variables
    LidarOdometryMode mode_ = LidarOdometryMode::IMU;
    bool first_gyr_ = true;
    bool first_acc_ = true;
    rclcpp::Time last_gyr_time_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
    rclcpp::Time last_acc_time_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
    bool acc_in_m_s2_ = true;
    bool invert_imu_ = false;
    double time_field_multiplier_ = 1e-9;
    bool absolute_time_ = false;
    double pc_scale_ = 1.0;
    std::set<int> broken_channels_;

    // Map Params
    double voxel_size_ = 0.2;
    double downsample_size_ = 0.4;
    size_t max_nb_pts_ = 4000;
    bool register_ = true;
    bool approximate_ = false;
    bool with_init_guess_ = true;
    bool use_edge_field_ = true;
    bool localization_ = false;
    bool key_framing_ = false;
    double key_framing_dist_thr_ = 1.0;
    double key_framing_rot_thr_ = 0.1;
    double key_framing_time_thr_ = 1.0;
    double key_framing_dist_cumulated_ = 0.0;
    double key_framing_time_cumulated_ = 0.0;
    bool first_map_update_ = true;
    bool pc_type_internal_ = true;
    double map_loss_scale_ = 0.5;
    std::string traj_path_;

    rclcpp::Time last_pc_time_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
    Mat4 current_pose_ = Mat4::Identity();
    Mat4 last_input_pose_ = Mat4::Identity();
    Mat4 init_guess_ = Mat4::Identity();

    std::atomic<int> pushed_pc_count_{0};
    std::atomic<int> processed_pc_count_{0};
    std::map<int64_t, Mat4> latest_global_odom_pose_;
    std::mutex pose_mutex_;


    void processBag(std::string bag_path)
    {
        rosbag2_cpp::Reader reader;
        int ctr = 0;
        try {
            reader.open(bag_path);
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Failed to open bag '%s': %s", bag_path.c_str(), e.what());
            rclcpp::shutdown();
            return;
        }

        rclcpp::Serialization<sensor_msgs::msg::PointCloud2> pc_serialization;
        rclcpp::Serialization<sensor_msgs::msg::Imu> imu_serialization;

        while (reader.has_next() && running_ && rclcpp::ok()) {
            
            while ((pushed_pc_count_.load() - processed_pc_count_.load()) > 20 && running_ && rclcpp::ok()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }

            auto bag_message = reader.read_next();
            const std::string& topic_name = bag_message->topic_name;

            if (topic_name == lidar_topic_) {
                if (ctr > 300) {
                    break;
                }
                rclcpp::SerializedMessage extracted_serialized_msg(*bag_message->serialized_data);
                auto msg = std::make_shared<sensor_msgs::msg::PointCloud2>();
                pc_serialization.deserialize_message(&extracted_serialized_msg, msg.get());
                processLidarMsg(msg);
            } 
            else if (topic_name == imu_acc_topic_ || topic_name == imu_gyr_topic_ || topic_name == "/imu/data") {
                rclcpp::SerializedMessage extracted_serialized_msg(*bag_message->serialized_data);
                auto msg = std::make_shared<sensor_msgs::msg::Imu>();
                imu_serialization.deserialize_message(&extracted_serialized_msg, msg.get());
                
                if (topic_name == imu_acc_topic_ || topic_name == "/imu/data") {
                    processAccMsg(msg);
                }
                if (topic_name == imu_gyr_topic_ || topic_name == "/imu/data") {
                    processGyrMsg(msg);
                }
            }
        }
        
        RCLCPP_INFO(this->get_logger(), "Finished reading bag. Waiting for LO queue to empty...");

        // Ensure all pushed point clouds have been processed by the LO thread and the map
        int timeout_counter = 0;
        int last_processed = processed_pc_count_.load();
        while(processed_pc_count_.load() < pushed_pc_count_.load() && running_ && rclcpp::ok()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            if (processed_pc_count_.load() == last_processed) {
                timeout_counter++;
                if (timeout_counter > 50) { // 5 seconds of no progress
                    RCLCPP_WARN(this->get_logger(), "Timeout waiting for LO queue. Some scans may have been dropped.");
                    break;
                }
            } else {
                last_processed = processed_pc_count_.load();
                timeout_counter = 0;
            }
        }

        RCLCPP_INFO(this->get_logger(), "All data processed. Shutting down offline processor.");
        rclcpp::shutdown();
    }

    void processLidarMsg(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        if( ((first_acc_ || first_gyr_) && (mode_ == LidarOdometryMode::IMU)) || (first_gyr_ && (mode_ == LidarOdometryMode::GYR)) ) {
            RCLCPP_WARN(this->get_logger(), "Received point cloud before IMU messages, ignoring the point cloud");
            return;
        }

        auto [incoming_pts, temp_has_intensity, temp_has_channel, is_2d] = pointCloud2MsgToPtsVec<double>(msg, time_field_multiplier_, true, broken_channels_, absolute_time_);
        auto incoming_pts_ptr = std::make_shared<std::vector<Pointd>>(std::move(incoming_pts));

        if(pc_scale_ != 1.0) {
            for(auto& pt : *incoming_pts_ptr) {
                pt.x *= pc_scale_;
                pt.y *= pc_scale_;
                pt.z *= pc_scale_;
            }
        }

        lidar_odometry_->setIs2D(is_2d);
        lidar_odometry_->addPc(incoming_pts_ptr, msg->header.stamp.nanosec + (uint64_t)msg->header.stamp.sec * 1000000000ull);
        
        pushed_pc_count_++;
    }

    void processAccMsg(const sensor_msgs::msg::Imu::SharedPtr msg)
    {
        uint64_t nanos = msg->header.stamp.nanosec + (uint64_t)msg->header.stamp.sec * 1000000000ull;
        rclcpp::Time header_time(nanos);
        if(first_acc_) {
            first_acc_ = false;
            last_acc_time_ = header_time - rclcpp::Duration::from_seconds(0.1);
        }
        if(header_time <= last_acc_time_) return;
        last_acc_time_ = header_time;

        Vec3 acc;
        acc << msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z;
        if(!acc_in_m_s2_) acc *= 9.81;
        if(invert_imu_) acc *= -1;
        lidar_odometry_->addAccSample(acc, nanos);
    }

    void processGyrMsg(const sensor_msgs::msg::Imu::SharedPtr msg)
    {
        uint64_t nanos = msg->header.stamp.nanosec + (uint64_t)msg->header.stamp.sec * 1000000000ull;
        rclcpp::Time header_time(nanos);
        if(first_gyr_) {
            first_gyr_ = false;
            last_gyr_time_ = header_time - rclcpp::Duration::from_seconds(0.1);
        }
        if(header_time <= last_gyr_time_) return;
        last_gyr_time_ = header_time;

        Vec3 gyr;
        gyr << msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z;
        if(invert_imu_) gyr *= -1;
        lidar_odometry_->addGyroSample(gyr, nanos);
    }


    // ====== Map Processing Logic ======

    int64_t getTimeNs(const rclcpp::Time& t) {
        return t.nanoseconds();
    }

    void updateMap(const std::vector<Pointd>& pts, const rclcpp::Time& time, const Mat4& trans)
    {
        bool add_to_map = false;

        if(first_map_update_) {
            last_pc_time_ = time;
            last_input_pose_ = trans;
            add_to_map = true;
        }

        if(time < last_pc_time_) {
            RCLCPP_WARN(this->get_logger(), "Time diff is negative, skipping point cloud");
            return;
        }

        if(!first_map_update_) {
            add_to_map = needMapUpdate(time, trans);
        }
        updateInitGuess(trans);

        if(add_to_map) {
            
            if(localization_ && first_map_update_) {
                std::vector<Pointd> downsampled_pts = downsamplePointCloud<double>(pts, downsample_size_, max_nb_pts_, true);
                std::lock_guard<std::mutex> lock(map_mutex_);
                current_pose_ = map_->registerPts(downsampled_pts, init_guess_, getTimeNs(time), true, 10.0, 10.0);
                current_pose_ = map_->registerPts(downsampled_pts, current_pose_, getTimeNs(time), true, 5.0, 10.0);
                current_pose_ = map_->registerPts(downsampled_pts, current_pose_, getTimeNs(time), true, 2.0, 10.0);
                current_pose_ = map_->registerPts(downsampled_pts, current_pose_, getTimeNs(time), approximate_, map_loss_scale_);
                init_guess_ = current_pose_;
            }
            else if(register_ && !first_map_update_) {
                std::vector<Pointd> downsampled_pts;
                if(use_edge_field_) downsampled_pts = downsamplePointCloudPerType<double>(pts, downsample_size_, max_nb_pts_);
                else downsampled_pts = downsamplePointCloud<double>(pts, downsample_size_, max_nb_pts_, false);

                std::lock_guard<std::mutex> lock(map_mutex_);
                if(!with_init_guess_) {
                    current_pose_ = map_->registerPts(downsampled_pts, current_pose_, getTimeNs(time), true, 10.0*map_loss_scale_);
                    init_guess_ = current_pose_;
                }
                current_pose_ = map_->registerPts(downsampled_pts, init_guess_, getTimeNs(time), approximate_, map_loss_scale_, 25);
                init_guess_ = current_pose_;
            }
            else {
                current_pose_ = trans;
            }

            {
                std::lock_guard<std::mutex> lock(map_mutex_);
                if(!localization_) {
                    map_->addPts(pts, current_pose_, getTimeNs(time));
                }
            }
        }

        logPoseToFile(traj_path_, init_guess_, time);

        last_input_pose_ = trans;
        last_pc_time_ = time;
        first_map_update_ = false;

        RCLCPP_INFO(this->get_logger(), "Processed PC time %f. Pose: %f, %f, %f", time.seconds(), current_pose_(0,3), current_pose_(1,3), current_pose_(2,3));
    }

    bool needMapUpdate(const rclcpp::Time& time, const Mat4& trans)
    {
        if(!key_framing_) return true;
        bool need_update = false;
        Mat4 delta_trans = last_input_pose_.inverse() * trans;
        if(key_framing_) {
            double time_diff = (time - last_pc_time_).seconds();
            key_framing_time_cumulated_ += time_diff;
            key_framing_dist_cumulated_ += delta_trans.block<3, 1>(0, 3).norm();
            if(key_framing_time_cumulated_ >= key_framing_time_thr_ || key_framing_dist_cumulated_ >= key_framing_dist_thr_) need_update = true;
            auto [dist, rot_diff] = distanceBetweenTransforms(current_pose_, init_guess_);
            if(dist >= key_framing_dist_thr_ || rot_diff >= key_framing_rot_thr_) need_update = true;
        }
        if(need_update) {
            key_framing_time_cumulated_ = 0.0;
            key_framing_dist_cumulated_ = 0.0;
        }
        return need_update;
    }

    void updateInitGuess(const Mat4& trans)
    {
        Mat4 delta_trans = last_input_pose_.inverse() * trans;
        init_guess_ = init_guess_*delta_trans;
    }

    void logPoseToFile(const std::string& path, const Mat4 & pose, const rclcpp::Time & time)
    {
        auto ns_epoch = time.nanoseconds();
        auto seconds = ns_epoch / 1000000000LL;
        auto nanoseconds = ns_epoch % 1000000000LL;
        std::ofstream trajectory_file(path, std::ios::out | std::ios::app);
        if (trajectory_file.is_open()) {
            Mat3 rot_mat = pose.block<3,3>(0,0);
            Eigen::Quaterniond q(rot_mat);
            trajectory_file << std::fixed << seconds << "."
                            << std::setfill('0') << std::setw(9) << nanoseconds << " "
                            << pose(0,3) << " "
                            << pose(1,3) << " "
                            << pose(2,3) << " "
                            << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
            trajectory_file.close();
        }
    }

    void createTrajectoryFile(const std::string& path)
    {
        std::ofstream trajectory_file(path, std::ios::out | std::ios::trunc);
        if (trajectory_file.is_open()) {
            trajectory_file.close();
        }
    }
};


int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<OfflineBoreasProcessorNode>();

    try {
        rclcpp::spin(node);
    } catch (const std::exception & e) {
        std::cout << "Exception: " << e.what() << std::endl;
    }

    // if thread naturally finished we can cleanly shut down here
    if (node) {
        node->shutdown();
    }

    rclcpp::shutdown();
    return 0;
}
