#pragma once

#include <torch/torch.h>

#include "data_parsers/base_parser.h"
#include "utils/sensor_utils/sensors.hpp"

namespace dataloader {
class DataLoader {
public:
  typedef std::shared_ptr<DataLoader> Ptr;
  explicit DataLoader(const std::string &dataset_path,
                      const std::filesystem::path &_config_path,
                      const int &_dataset_type = 0,
                      const torch::Device &_device = torch::kCPU,
                      const bool &_preload = false,
                      const float &_res_scale = 1.0,
                      const sensor::Sensors &_sensor = sensor::Sensors(),
                      const int &_ds_pt_num = 1e5,
                      const float &_max_time_diff_camera_and_pose = 0.0f,
                      const float &_max_time_diff_lidar_and_pose = 0.0f);

  // Base paths.
  // std::string calib_path_, poses_path_;
  torch::Device device_ = torch::kCPU;

  dataparser::DataParser::Ptr dataparser_ptr_;

  bool get_next_data(int idx, torch::Tensor &_pose, PointCloudT &_points);

  bool get_next_data(int idx, torch::Tensor &_pose, DepthSamples &_depth_rays,
                     ColorSamples &_color_rays,
                     const torch::Device &_device = torch::Device(torch::kCPU));

  bool get_next_data(int idx, torch::Tensor &_pose, DepthSamples &_depth_rays,
                     const torch::Device &_device = torch::Device(torch::kCPU));
  bool get_next_data(int idx, torch::Tensor &_pose, ColorSamples &_color_rays,
                     const torch::Device &_device = torch::Device(torch::kCPU));

  torch::Tensor get_pose(int idx, const int &pose_type = 0);

  int export_image(std::filesystem::path &gs_sparse_path,
                   std::filesystem::path &gs_path, int data_type,
                   std::string filename, bool bin, uint32_t camera_id,
                   bool eval, bool llff, std::string prefix = "",
                   int prefix_num = 0);
  void export_as_colmap_format(bool bin = true, bool llff = false);
  void export_as_colmap_format_for_nerfstudio(bool bin = true);

private:
  // torch::Tensor Tr;

  // std::vector<torch::Tensor> poses_vec_;
  // std::vector<std::string> depth_name_vec_;

  int dataset_type_;

  // bool load_calib();
  // bool load_poses();

  // bool load_depth_list();
};
} // namespace dataloader