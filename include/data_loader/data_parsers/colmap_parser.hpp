#pragma once
#include "base_parser.h"
#include "utils/coordinates.h"
#include <pcl/io/ply_io.h>

namespace dataparser {
struct Colmap : DataParser {

  std::filesystem::path depth_pose_path_, mask_file_;
  int color_pose_type_, depth_pose_type_ = 0;
  bool color_pose_w2c_ = false;

  explicit Colmap(const std::filesystem::path &_dataset_path,
                  const std::filesystem::path &_config_path,
                  const torch::Device &_device = torch::kCPU,
                  const bool &_preload = true, const float &_res_scale = 1.0,
                  const sensor::Sensors &_sensor = sensor::Sensors(),
                  const int &_ds_pt_num = 1e5,
                  const float &_max_time_diff_camera_and_pose = 0.0f,
                  const float &_max_time_diff_lidar_and_pose = 0.0f,
                  bool is_father = false)
      : DataParser(_dataset_path, _device, _preload, _res_scale,
                   coords::SystemType::OpenCV, _sensor, _ds_pt_num,
                   _max_time_diff_camera_and_pose,
                   _max_time_diff_lidar_and_pose) {
    dataset_name_ = dataset_path_.filename();

    auto config = read_params(_dataset_path, _config_path);
    color_path_ = config.color_path;
    color_pose_path_ = config.color_pose_path;
    depth_path_ = config.depth_path;
    depth_pose_path_ = config.depth_pose_path;
    mask_file_ = dataset_path_ / "images/right_undistorded_mask.jpg";

    depth_type_ = config.depth_type;
    color_pose_type_ = config.color_pose_type;
    color_pose_w2c_ = config.color_pose_w2c;
    depth_pose_type_ = config.depth_pose_type;

    load_intrinsics();
    load_data();

    int skip_first_num = 0;
    post_process(skip_first_num);
  }

  void load_data() override {
    time_stamps_ = torch::Tensor(); // reset time_stamps

    // export undistorted images

    if (!std::filesystem::exists(color_path_)) {
      throw std::runtime_error("color_path_ does not exist: " +
                               color_path_.string());
    }
    if (!std::filesystem::exists(color_pose_path_)) {
      throw std::runtime_error("pose_path_ does not exist: " +
                               color_pose_path_.string());
    }
    if (!std::filesystem::exists(depth_pose_path_)) {
      throw std::runtime_error("depth_pose_path_ does not exist: " +
                               depth_pose_path_.string());
    }
    if (!std::filesystem::exists(depth_path_)) {
      throw std::runtime_error("depth_path_ does not exist: " +
                               depth_path_.string());
    }
    auto color_info = load_poses(color_pose_path_, false, color_pose_type_,
                                 true, "", color_pose_w2c_);
    color_poses_ = std::get<0>(color_info);
    raw_color_filelists_ = std::get<2>(color_info);
    std::cout << "Loaded " << color_poses_.size(0) << " color poses\n";
    TORCH_CHECK(color_poses_.size(0) > 0);
    load_colors(".jpg", "", false, true);
    std::cout << "Loaded " << raw_color_filelists_.size() << " color images\n";
    TORCH_CHECK(color_poses_.size(0) == raw_color_filelists_.size());

    if (std::filesystem::exists(mask_file_)) {
      mask = get_color_image(mask_file_, 0) > 0;
    }

    depth_poses_ =
        std::get<0>(load_poses(depth_pose_path_, false, depth_pose_type_));
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