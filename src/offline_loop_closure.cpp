#include "cxxopts/include/cxxopts.hpp"
#include "lice/submap_bundle_adjustment.h"

int main(int argc, char** argv)
{

    // Define command-line options
    cxxopts::Options options("Offline loop closure detection and correction", "A tool to perform offline loop closure detection, pose graph optimization, and bundle adjustment on lidar submaps");
    options.add_options()
        ("d,data_folder", "Path to the data folder with map.ply or ", cxxopts::value<std::string>())
        ("odom_pos_std", "Standard deviation of odometry position [m]", cxxopts::value<double>()->default_value("0.2"))
        ("odom_rot_std", "Standard deviation of odometry rotation [rad]", cxxopts::value<double>()->default_value("1.0"))
        ("loop_pos_std", "Standard deviation of loop closure position [m]", cxxopts::value<double>()->default_value("0.2"))
        ("loop_rot_std", "Standard deviation of loop closure rotation [rad]", cxxopts::value<double>()->default_value(std::to_string(1.0 * M_PI / 180.0)))
        ("loop_loss_scale_pos", "Loss function scale for loop closures", cxxopts::value<double>()->default_value("1.0"))
        ("loop_loss_scale_rot", "Loss function scale for loop closures", cxxopts::value<double>()->default_value(std::to_string(1.0 * M_PI / 180.0)))
        ("ram_efficient", "Use RAM efficient mode for bundle adjustment (much slower)")
        ("h,help", "Print help");


    auto result = options.parse(argc, argv);

    // Handle help option
    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        return 0;
    }

    // Retrieve option values
    SubmapBundleAdjustmentOptions sba_options;
    std::string data_folder = result["data_folder"].as<std::string>();
    sba_options.odom_pos_std = result["odom_pos_std"].as<double>();
    sba_options.odom_rot_std = result["odom_rot_std"].as<double>();
    sba_options.loop_pos_std = result["loop_pos_std"].as<double>();
    sba_options.loop_rot_std = result["loop_rot_std"].as<double>();
    sba_options.loop_loss_scale_pos = result["loop_loss_scale_pos"].as<double>();
    sba_options.loop_loss_scale_rot = result["loop_loss_scale_rot"].as<double>();

    if(result.count("ram_efficient"))
    {
        sba_options.ram_efficient = true;
    }

    SubmapBundleAdjustment bundle_adjustment(data_folder, sba_options);
    bundle_adjustment.refineLoopTransforms();
    bundle_adjustment.poseGraphOptimization();

    return 0;
}
