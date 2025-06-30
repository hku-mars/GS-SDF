#pragma once
#include "base_parser.h"
#include "utils/coordinates.h"
#include "utils/sensor_utils/cameras.hpp"

namespace dataparser {
struct Replica : DataParser {
  explicit Replica(const std::string &_dataset_path,
                   const torch::Device &_device = torch::kCPU,
                   const bool &_preload = true, const float &_res_scale = 1.0,
                   const int &_ds_pt_num = 1e5)
      : DataParser(_dataset_path, _device, _preload, _res_scale,
                   coords::SystemType::OpenCV, sensor::Sensors(), _ds_pt_num) {
    color_pose_path_ = dataset_path_ / "traj.txt";
    color_path_ = dataset_path_ / "results";
    depth_path_ = color_path_;

    eval_pose_path_ = dataset_path_ / "eval" / "traj.txt";
    eval_color_path_ = dataset_path_ / "eval" / "results";
    eval_depth_path_ = eval_color_path_;

    gt_mesh_path_ = dataset_path_.parent_path() / "cull_replica_mesh" /
                    (dataset_path_.filename().string() + "_culled.ply");
    // gt_mesh_path_ = dataset_path_.parent_path() / "cull_replica_mesh" /
    //                 (dataset_path_.filename().string() + ".ply");
    // gt_mesh_path_ = dataset_path_.parent_path() /
    //                 (dataset_path_.filename().string() + "_mesh.ply");

    load_intrinsics();
    if (std::filesystem::exists(color_pose_path_)) {
      load_data();
    } else {
      std::cout << "pose_path_ does not exist: " << color_pose_path_ << std::endl;
    }
  }

  std::filesystem::path gt_mesh_path_;

  void load_data() override {
    depth_poses_ = std::get<0>(load_poses(color_pose_path_, false, 1));
    color_poses_ = depth_poses_;
    TORCH_CHECK(depth_poses_.size(0) > 0);

    load_colors(".jpg", "frame", false, false);
    TORCH_CHECK(depth_poses_.size(0) == raw_color_filelists_.size());
    load_depths(".png", "depth", false, false);
    TORCH_CHECK(raw_color_filelists_.size() == raw_depth_filelists_.size());

    load_eval_data();
  }

  void load_eval_data() override {
    eval_color_poses_ = std::get<0>(load_poses(eval_pose_path_, false, 1));
    if (eval_color_poses_.size(0) <= 0) {
      std::cout << "eval_color_poses_.size(0) <= 0" << std::endl;
      return;
    }

    load_colors(".jpg", "frame", true);
    TORCH_CHECK(eval_color_poses_.size(0) == eval_color_filelists_.size());
    load_depths(".png", "depth", true);
    TORCH_CHECK(eval_color_filelists_.size() == eval_depth_filelists_.size());

    eval_depth_poses_ = eval_color_poses_;
  }

  void load_intrinsics() override {
    // Replica/cam_params.json
    sensor_.camera.width = 1200;
    sensor_.camera.height = 680;
    sensor_.camera.fx = 600.0;
    sensor_.camera.fy = 600.0;
    sensor_.camera.cx = 599.5;
    sensor_.camera.cy = 339.5;
    depth_scale_inv_ = 1.0 / 6553.5;
  }

  std::string get_gt_mesh_path() override { return gt_mesh_path_.string(); }
};
} // namespace dataparser