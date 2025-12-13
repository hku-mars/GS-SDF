#pragma once
#include "base_parser.h"
#include "utils/coordinates.h"

namespace dataparser {
struct NewerCollege : DataParser {
  explicit NewerCollege(
      const std::string &_dataset_path,
      const torch::Device &_device = torch::kCPU, const bool &_preload = true,
      const int &_dataset_system_type = coords::SystemType::OpenCV,
      const int &_ds_pt_num = 1e5)
      : DataParser(_dataset_path, _device, _preload, _dataset_system_type,
                   sensor::Sensors(), _ds_pt_num) {
    depth_type_ = DepthType::Image;
    // Deprecated: no color images

    std::cout << dataset_path_ << std::endl;
    // input _dataset_path should be end with .bag:
    // NewerCollegeDataset/2021-ouster-os0-128-alphasense/collection 1 - newer
    // college/2021-07-01-10-37-38-quad-easy.bag
    auto bag_name = dataset_path_.filename().string();
    // extract quad-easy from 2021-07-01-10-37-38-quad-easy.bag
    bag_name = bag_name.substr(0, bag_name.find(".bag"));
    bag_name = bag_name.substr(bag_name.find_last_of("0123456789") + 2);
    std::cout << bag_name << std::endl;

    dataset_path_ = dataset_path_.parent_path();
    std::cout << dataset_path_ << std::endl;

    pose_path_ = dataset_path_ / "ground_truth";
    color_path_ = dataset_path_ / "results";
    depth_path_ = color_path_;

    gt_mesh_path_ = dataset_path_.parent_path() / "cull_replica_mesh" /
                    (dataset_path_.filename().string() + ".ply");
    // gt_mesh_path_ = dataset_path_.parent_path() /
    //                 (dataset_path_.filename().string() + "_mesh.ply");

    load_intrinsics();
  }

  std::filesystem::path gt_mesh_path_;

  void load_data() override {
    depth_poses_ = std::get<0>(load_poses(pose_path_, false, 1));
    TORCH_CHECK(depth_poses_.size(0) > 0);

    load_colors(".jpg", "frame", false, true);
    TORCH_CHECK(depth_poses_.size(0) == color_filelists_.size());
    load_depths(depth_type_, "depth", false, true);
    TORCH_CHECK(color_filelists_.size() == depth_filelists_.size());
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