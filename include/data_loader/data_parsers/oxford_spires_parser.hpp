#pragma once
#include "base_parser.h"
#include "utils/coordinates.h"
#include "utils/sensor_utils/cameras.hpp"
#include <pcl/io/ply_io.h>

namespace dataparser {
struct Spires : DataParser {
  explicit Spires(const std::filesystem::path &_dataset_path,
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
    // export undistorted images
    color_pose_path_ = dataset_path_ / "color_poses.txt";
    depth_pose_path_ = dataset_path_ / "depth_poses.txt";
    color_path_ = dataset_path_ / "undistorted_images";
    depth_path_ = dataset_path_ / "depths"; // origin data
    // /media/chrisliu/T9/Datasets/Oxford_Spires_Dataset/bodleian_library/02
    // └── 02
    //     ├── gt-tum.txt
    //     ├── images
    //     │   ├── cam0
    //     │   ├── cam1
    //     │   └── cam2
    //     └── lidar-clouds
    dataset_name_ = dataset_path_.filename();

    // pose_path_ = dataset_path_ / "gt-tum.txt";
    // color_path_ = dataset_path_ / "images" / "cam0";
    depth_type_ = DepthType::PCD;
    load_intrinsics();
    load_data();
  }

  torch::Tensor T_C0_L, T_C0_C2;

  std::filesystem::path depth_pose_path_;
  void load_data() override {
    if (!std::filesystem::exists(color_pose_path_) ||
        !std::filesystem::exists(depth_pose_path_) ||
        !std::filesystem::exists(color_path_) ||
        !std::filesystem::exists(depth_path_)) {
      color_pose_path_ = dataset_path_ / "gt-tum.txt";
      color_path_ = dataset_path_ / "images" / "cam0";
      depth_path_ = dataset_path_ / "lidar-clouds";

      auto pose_data = load_poses(color_pose_path_, false, 3);
      auto T_W_B = std::get<0>(pose_data);
      auto T_W_L = T_W_B.matmul(sensor_.T_B_L);
      time_stamps_ = std::get<1>(pose_data);
      TORCH_CHECK(T_W_L.size(0) > 0);
      // auto T_W_C0 =
      //     coords::change_world_system(T_C0_C0, coords::SystemType::Kitti);
      color_poses_ = T_W_L.matmul(sensor_.T_C_L.inverse());
      depth_poses_ = T_W_L;

      load_colors(".jpg", "", false, false);
      std::cout << "color_poses: " << color_poses_.size(0) << " color_files"
                << raw_color_filelists_.size() << std::endl;
      load_depths(".pcd", "", false, false);
      std::cout << "depth_poses: " << depth_poses_.size(0)
                << " depth_files: " << raw_depth_filelists_.size() << std::endl;

      // export undistorted images
      color_path_ = dataset_path_ / "undistorted_images";
      std::filesystem::create_directories(color_path_);
      color_pose_path_ = dataset_path_ / "color_poses.txt";
      std::ofstream color_pose_file(color_pose_path_);
      for (int i = 0; i < raw_color_filelists_.size(); i++) {
        auto color_image = get_image_cv_mat(i);
        auto undistorted_img_path =
            color_path_ / raw_color_filelists_[i].filename();
        cv::imwrite(undistorted_img_path, color_image);

        auto T_W_C = color_poses_[i];
        for (int i = 0; i < 4; i++) {
          for (int j = 0; j < 4; j++) {
            color_pose_file << T_W_C[i][j].item<float>() << " ";
          }
          color_pose_file << "\n";
        }
      }
      // export align pose depth
      depth_path_ = dataset_path_ / "depths";
      std::filesystem::create_directories(depth_path_);
      std::ofstream depth_pose_file(depth_pose_path_);
      for (int i = 0; i < raw_depth_filelists_.size(); i++) {
        // copy depth file to undistorted_images
        std::filesystem::copy_file(
            raw_depth_filelists_[i],
            depth_path_ / raw_depth_filelists_[i].filename(),
            std::filesystem::copy_options::overwrite_existing);

        auto T_W_L = depth_poses_[i];
        for (int i = 0; i < 4; i++) {
          for (int j = 0; j < 4; j++) {
            depth_pose_file << T_W_L[i][j].item<float>() << " ";
          }
          depth_pose_file << "\n";
        }
      }
    }

    time_stamps_ = torch::Tensor(); // reset time_stamps

    color_poses_ = std::get<0>(load_poses(color_pose_path_, false, 0));
    TORCH_CHECK(color_poses_.size(0) > 0);
    depth_poses_ = std::get<0>(load_poses(depth_pose_path_, false, 0));
    TORCH_CHECK(depth_poses_.size(0) > 0);

    load_colors(".jpg", "", false, true);
    TORCH_CHECK(color_poses_.size(0) == raw_color_filelists_.size());
    load_depths(".pcd", "", false, true);
    TORCH_CHECK(depth_poses_.size(0) == raw_depth_filelists_.size());
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