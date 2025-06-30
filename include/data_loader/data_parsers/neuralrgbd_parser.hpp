#pragma once
#include "base_parser.h"
#include "utils/coordinates.h"
#include "utils/sensor_utils/cameras.hpp"

namespace dataparser {
struct NeuralRGBD : DataParser {
  explicit NeuralRGBD(const std::string &_dataset_path,
                      const torch::Device &_device = torch::kCPU,
                      const bool &_preload = true,
                      const float &_res_scale = 1.0, const int &_depth_type = 0,
                      const int &_dataset_system_type = 2,
                      const int &_ds_pt_num = 0)
      : DataParser(_dataset_path, _device, _preload, _res_scale,
                   _dataset_system_type) {
    color_pose_path_ = dataset_path_ / "poses.txt";
    color_path_ = dataset_path_ / "images";
    switch (_depth_type) {
    case 0:
      depth_path_ = dataset_path_ / "depth";
      break;
    case 1:
      depth_path_ = dataset_path_ / "depth_filtered";
      break;
    case 2:
      depth_path_ = dataset_path_ / "depth_with_noise";
      break;
    default:
      throw std::runtime_error("Invalid depth type");
    }

    focal_length_path_ = dataset_path_ / "focal.txt";

    // gt_mesh_path_ = dataset_path_ / "gt_mesh.ply";
    gt_mesh_path_ = dataset_path_ / "gt_mesh_culled.ply";

    load_intrinsics();
  }

  std::filesystem::path focal_length_path_;
  std::filesystem::path gt_mesh_path_;

  void load_data() override {
    depth_poses_ = std::get<0>(load_poses(color_pose_path_, false, 0));
    TORCH_CHECK(depth_poses_.size(0) > 0);
    /* // https://github.com/PRBonn/semantic-kitti-api/issues/115
    // https://github.com/PRBonn/kiss-icp/issues/156#issuecomment-1551695713
    https://github.com/PRBonn/semantic-kitti-api/issues/133
    poses_ = T_S_B_.matmul(poses_.matmul(T_B_S_)); */
    depth_poses_ =
        coords::change_world_system(depth_poses_, dataset_system_type_);
    depth_poses_ =
        coords::change_camera_system(depth_poses_, dataset_system_type_);
    color_poses_ = depth_poses_;

    load_colors(".png", "img", false, true);
    TORCH_CHECK(depth_poses_.size(0) == raw_color_filelists_.size());
    load_depths(".png", "depth", false, true);
    TORCH_CHECK(raw_color_filelists_.size() == raw_depth_filelists_.size());
  }

  virtual void load_focal_length() {
    if (!std::filesystem::exists(focal_length_path_)) {
      throw std::runtime_error("Focal Length file does not exist");
    }
    std::ifstream import_file(focal_length_path_, std::ios::in);
    if (!import_file) {
      throw std::runtime_error("Could not open focal length file");
    }
    std::string line;
    std::getline(import_file, line);
    sensor_.camera.focal_length_ = std::stof(line);
    import_file.close();
  }

  void load_intrinsics() override {
    sensor_.camera.width = 640;
    sensor_.camera.height = 480;

    load_focal_length();
    sensor_.camera.fx = sensor_.camera.focal_length_;
    sensor_.camera.fy = sensor_.camera.focal_length_;
    sensor_.camera.cx = 0.5 * (sensor_.camera.width - 1);
    sensor_.camera.cy = 0.5 * (sensor_.camera.height - 1);
  }

  std::string get_gt_mesh_path() override { return gt_mesh_path_.string(); }
};
} // namespace dataparser