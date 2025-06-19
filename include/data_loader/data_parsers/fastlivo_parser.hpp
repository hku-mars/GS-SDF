#pragma once
#include "data_loader/data_parsers/rosbag_parser.hpp"
#include "utils/coordinates.h"
#include "utils/sensor_utils/cameras.hpp"
#include "utils/sensor_utils/sensors.hpp"
#include <pcl/io/ply_io.h>

namespace dataparser {
struct Fastlivo : Rosbag {
  explicit Fastlivo(const std::filesystem::path &_bag_path,
                    const torch::Device &_device = torch::kCPU,
                    const bool &_preload = true, const float &_res_scale = 1.0,
                    const sensor::Sensors &_sensor = sensor::Sensors(),
                    const int &_ds_pt_num = 1e5)
      : Rosbag(_bag_path, _device, _preload, _res_scale,
               coords::SystemType::OpenCV, _sensor, _ds_pt_num) {
    depth_type_ = DepthType::PLY;

    dataset_name_ = bag_path_.filename();
    dataset_name_ = dataset_name_.replace_extension();

    pose_topic = "/aft_mapped_to_init";
    color_topic = "/origin_img";
    depth_topic = "/cloud_registered_body";

    load_intrinsics();

    load_data();
    if (std::filesystem::exists(pose_path_)) {
    } else {
      std::cout << "pose_path_ does not exist: " << pose_path_ << std::endl;
    }
  }

  void load_intrinsics() override {
    // auto scale = 0.5f; // output image from Fastlivo is 640x512
    auto scale = 1.0f; // output image from Fastlivo is 1280x1024
    sensor_.camera.width = scale * sensor_.camera.width;
    sensor_.camera.height = scale * sensor_.camera.height;
    // HKU
    sensor_.camera.fx = scale * sensor_.camera.fx;
    sensor_.camera.fy = scale * sensor_.camera.fy;
    sensor_.camera.cx = scale * sensor_.camera.cx;
    sensor_.camera.cy = scale * sensor_.camera.cy;
  }
};
} // namespace dataparser