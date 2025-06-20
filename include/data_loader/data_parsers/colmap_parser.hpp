#pragma once
#include "base_parser.h"
#include "utils/coordinates.h"
#include <pcl/io/ply_io.h>

namespace dataparser {
struct Colmap : DataParser {
  explicit Colmap(const std::filesystem::path &_dataset_path,
                  const torch::Device &_device = torch::kCPU,
                  const bool &_preload = true, const float &_res_scale = 1.0,
                  const sensor::Sensors &_sensor = sensor::Sensors(),
                  const int &_ds_pt_num = 1e5,
                  const float &_max_time_diff_camera_and_pose = 0.0f,
                  const float &_max_time_diff_lidar_and_pose = 0.0f)
      : DataParser(_dataset_path, _device, _preload, _res_scale,
                   coords::SystemType::OpenCV, _sensor, _ds_pt_num,
                   _max_time_diff_camera_and_pose,
                   _max_time_diff_lidar_and_pose) {
    dataset_name_ = dataset_path_.filename();

    depth_type_ = DepthType::PCD;
    load_intrinsics();
    load_data();

    int skip_first_num = 0;
    post_process(skip_first_num);
  }

  std::filesystem::path depth_pose_path_;
  void load_data() override {
    time_stamps_ = torch::Tensor(); // reset time_stamps

    // export undistorted images
    color_path_ = dataset_path_ / "colmap/images";
    auto image_prefix = color_path_.stem();
    pose_path_ = dataset_path_ / "colmap/postrior_lidar/images.txt";
    depth_path_ = dataset_path_ / "depths";
    depth_pose_path_ = dataset_path_ / "depths/lidar_pose.txt";
    auto mask_file = dataset_path_ / "images/right_undistorded_mask.jpg";

    if (!std::filesystem::exists(color_path_)) {
      throw std::runtime_error("color_path_ does not exist: " +
                               color_path_.string());
    }
    if (!std::filesystem::exists(pose_path_)) {
      throw std::runtime_error("pose_path_ does not exist: " +
                               pose_path_.string());
    }
    if (!std::filesystem::exists(depth_pose_path_)) {
      throw std::runtime_error("depth_pose_path_ does not exist: " +
                               depth_pose_path_.string());
    }
    if (!std::filesystem::exists(depth_path_)) {
      throw std::runtime_error("depth_path_ does not exist: " +
                               depth_path_.string());
    }
    auto color_info = load_poses(pose_path_, false, 4, true, "", true);
    color_poses_ = std::get<0>(color_info);
    raw_color_filelists_ = std::get<2>(color_info);
    std::cout << "Loaded " << color_poses_.size(0) << " color poses\n";
    TORCH_CHECK(color_poses_.size(0) > 0);
    load_colors(".jpg", "", false, true);
    std::cout << "Loaded " << raw_color_filelists_.size() << " color images\n";
    TORCH_CHECK(color_poses_.size(0) == raw_color_filelists_.size());

    if (std::filesystem::exists(mask_file)) {
      mask = get_color_image(mask_file, 0) > 0;
    }

    depth_poses_ = std::get<0>(load_poses(depth_pose_path_, false, 5));
    std::cout << "Loaded " << depth_poses_.size(0) << " depth poses\n";
    TORCH_CHECK(depth_poses_.size(0) > 0);

    load_depths(".pcd", "", false, true);
    std::cout << "Loaded " << raw_depth_filelists_.size() << " depths\n";
    TORCH_CHECK(depth_poses_.size(0) == raw_depth_filelists_.size());
  }

  std::vector<at::Tensor> get_distance_ndir_zdirn(const int &idx) override {
    /**
     * @description:
     * @return {distance, ndir, dir_norm}, where ndir.norm = 1;
               {[height width 1], [height width 3], [height width 1]}
     */

    auto pointcloud = get_depth_image(idx);
    // [height width 1]
    auto distance = pointcloud.norm(2, -1, true);
    auto ndir = pointcloud / distance;
    return {distance, ndir, distance};
  }
};
} // namespace dataparser