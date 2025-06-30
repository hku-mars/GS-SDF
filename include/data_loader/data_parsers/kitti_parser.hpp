#pragma once
#include "base_parser.h"
#include "utils/coordinates.h"
#include "utils/sensor_utils/cameras.hpp"
#include <pcl/io/ply_io.h>

namespace dataparser {
struct Kitti : DataParser {
  explicit Kitti(const std::filesystem::path &_dataset_lidar_path,
                 const torch::Device &_device = torch::kCPU,
                 const bool &_preload = true, const float &_res_scale = 1.0,
                 const int &_ds_pt_num = 1e5)
      : DataParser(_dataset_lidar_path, _device, _preload, _res_scale,
                   coords::SystemType::OpenCV, sensor::Sensors(), _ds_pt_num) {
    depth_type_ = DepthType::BIN;

    // KITTI/data_odometry_velodyne/dataset/sequences/00
    dataset_name_ = _dataset_lidar_path.filename();
    // /media/chrisliu/T9/Datasets/KITTI/data_odometry_poses

    auto base_path = (dataset_path_ / "../../../..").lexically_normal();
    calib_path_ = base_path / "data_odometry_calib/dataset/sequences" /
                  dataset_name_ / "calib.txt";
    color_pose_path_ = base_path / "data_odometry_poses/dataset/poses" /
                 (dataset_name_.string() + ".txt");
    color_path_ = base_path / "data_odometry_color/dataset/sequences" /
                  dataset_name_ / "image_2";
    depth_path_ = base_path / "data_odometry_velodyne/dataset/sequences" /
                  dataset_name_ / "velodyne";
    load_intrinsics();
  }

  torch::Tensor T_C0_L, T_C0_C2;

  void load_data() override {

    auto T_C0_C0 = std::get<0>(load_poses(color_pose_path_, false, 2));
    TORCH_CHECK(T_C0_C0.size(0) > 0);
    auto T_W_C0 =
        coords::change_world_system(T_C0_C0, coords::SystemType::Kitti);
    color_poses_ = T_W_C0.matmul(T_C0_C2);
    depth_poses_ = T_W_C0.matmul(T_C0_L);

    load_colors(".png", "", false, true);
    TORCH_CHECK(color_poses_.size(0) == raw_color_filelists_.size());
    load_depths(".bin", "", false, true);
    TORCH_CHECK(depth_poses_.size(0) == raw_depth_filelists_.size());
  }

  void load_intrinsics() override {
    // open calibration file
    std::ifstream calib_file(calib_path_);
    if (!calib_file) {
      throw std::runtime_error("Could not open calibration file");
    }

    std::string line;
    T_C0_L = torch::eye(4);
    T_C0_C2 = torch::eye(4);
    while (std::getline(calib_file, line)) {
      std::istringstream iss(line);
      std::string token;
      iss >> token;
      if (token == "P2:") {
        int index = 0;
        torch::Tensor baseline = torch::zeros(3);
        while (iss >> token) {
          if (index == 0) {
            sensor_.camera.fx = std::stof(token);
          } else if (index == 2) {
            sensor_.camera.cx = std::stof(token);
          } else if (index == 3) {
            T_C0_C2[0][3] = -std::stof(token) / sensor_.camera.fx;
          } else if (index == 5) {
            sensor_.camera.fy = std::stof(token);
          } else if (index == 6) {
            sensor_.camera.cy = std::stof(token);
          } else if (index == 7) {
            T_C0_C2[1][3] = -std::stof(token) / sensor_.camera.fy;
          } else if (index == 11) {
            T_C0_C2[2][3] = -std::stof(token);
          }
          index++;
        }
      } else if (token == "Tr:") {
        int i = 0;
        int j = 0;
        float value;
        while (iss >> value) {
          T_C0_L[i][j] = value;
          j++;
          if ((j % 4) == 0) {
            i++;
            j = 0;
          }
        }
      }
    }
    sensor_.camera.width = 1241;
    sensor_.camera.height = 376;
    depth_scale_inv_ = 1.0;
    // print out cameras
    std::cout << "fx: " << sensor_.camera.fx << ", fy: " << sensor_.camera.fy
              << ", cx: " << sensor_.camera.cx << ", cy: " << sensor_.camera.cy
              << "\n";
    std::cout << "T_C0_L:\n" << T_C0_L << "\n";
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