#pragma once
#include "data_loader/data_parsers/rosbag_parser.hpp"
#include "utils/coordinates.h"
#include "utils/sensor_utils/cameras.hpp"
#include <pcl/io/ply_io.h>

namespace dataparser {
struct R3live : Rosbag {
  explicit R3live(const std::filesystem::path &_bag_path,
                  const torch::Device &_device = torch::kCPU,
                  const bool &_preload = true, const float &_res_scale = 1.0,
                  const int &_dataset_system_type = coords::SystemType::OpenCV)
      : Rosbag(_bag_path, _device, _preload, _res_scale, _dataset_system_type) {
    depth_type_ = DepthType::PLY;

    dataset_name_ = bag_path_.filename();
    dataset_name_ = dataset_name_.replace_extension();

    pose_topic = "/Odometry";
    color_topic = "/camera/image_color/compressed";
    depth_topic = "/cloud_registered_body";

    // camera to imu
    sensor_.T_B_C =
        torch::tensor({{-0.00113207, -0.0158688, 0.999873, 0.050166},
                       {-0.9999999, -0.000486594, -0.00113994, 0.0474116},
                       {0.000504622, -0.999874, -0.0158682, -0.0312415},
                       {0.0, 0.0, 0.0, 1.0}},
                      torch::kFloat);
    // lidar to imu
    sensor_.T_B_L = torch::tensor({{1.0, 0.0, 0.0, 0.04165},
                                   {0.0, 1.0, 0.0, 0.02326},
                                   {0.0, 0.0, 1.0, -0.0284},
                                   {0.0, 0.0, 0.0, 1.0}},
                                  torch::kFloat);
    load_intrinsics();
  }

  void load_intrinsics() override {
    // Replica/cam_params.json
    sensor_.camera.width = 1200;
    sensor_.camera.height = 1024;
    sensor_.camera.fx = 863.4241;
    sensor_.camera.fy = 863.4171;
    sensor_.camera.cx = 640.6808;
    sensor_.camera.cy = 518.3392;
    depth_scale_inv_ = 1.0;

    sensor_.camera.set_distortion(-0.1080, 0.1050, -1.2872e-04, 5.7923e-05,
                                  -0.0222);
  }
};
} // namespace dataparser