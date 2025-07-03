#pragma once

#include "utils/sensor_utils/cameras.hpp"
#include "utils/sensor_utils/sensors.hpp"
#include "utils/utils.h"
#include <dirent.h>
#include <filesystem>

#include <opencv2/opencv.hpp>
#include <torch/torch.h>

namespace dataparser {

enum DataType {
  RawColor = 0,
  RawDepth = 1,
  TrainColor = 2,
  TrainDepth = 3,
  EvalColor = 4,
  EvalDepth = 5,
  TestColor = 6,
  TestDepth = 7
};

enum DepthType { Image = 0, PLY = 1, BIN = 2, PCD = 3 };

std::vector<std::filesystem::path>
read_filelists(const std::filesystem::path &directory,
               const std::string &prefix,
               const std::string &extension = ".ply");

void sort_filelists(std::vector<std::filesystem::path> &filists);

void load_file_list(const std::string &dir_path,
                    std::vector<std::filesystem::path> &out_filelsits,
                    const std::string &prefix,
                    const std::string &file_extension = ".ply");

struct DataConfig {
  std::filesystem::path color_path, color_pose_path, depth_path,
      depth_pose_path;
  DepthType depth_type;
  int color_pose_type, depth_pose_type;
  bool color_pose_w2c = false;
};

DataConfig read_params(const std::filesystem::path &_dataset_path,
                       const std::filesystem::path &_config_path);

struct DataParser {
  typedef std::shared_ptr<DataParser> Ptr;
  explicit DataParser(const std::string &_dataset_path,
                      const torch::Device &_device = torch::kCPU,
                      const bool &_preload = true,
                      const float &_res_scale = 1.0,
                      const int &_dataset_system_type = 0,
                      const sensor::Sensors &_sensor = sensor::Sensors(),
                      const int &_ds_pt_num = 1e5,
                      const float &_max_time_diff_camera_and_pose = 0.0f,
                      const float &_max_time_diff_lidar_and_pose = 0.0f)
      : dataset_path_(_dataset_path), device_(_device), preload_(_preload),
        res_scale_(_res_scale), dataset_system_type_(_dataset_system_type),
        sensor_(_sensor), ds_pt_num_(_ds_pt_num),
        max_time_diff_camera_and_pose_(_max_time_diff_camera_and_pose),
        max_time_diff_lidar_and_pose_(_max_time_diff_lidar_and_pose) {
          // you are supposed to call load_data() in the derived class
        };

  std::filesystem::path dataset_path_, dataset_name_;
  std::filesystem::path color_pose_path_, calib_path_, color_path_, depth_path_;

  std::filesystem::path eval_pose_path_, eval_color_path_, eval_depth_path_;

  torch::Device device_ = torch::kCPU;
  torch::Tensor train_color_;     // [N, H, W, 3]
  DepthSamples train_depth_pack_; // [N]
  torch::Tensor depth_poses_, train_depth_poses_, color_poses_,
      train_color_poses_;                             // [N, 3, 4]
  torch::Tensor time_stamps_;                         // [N]
  torch::Tensor eval_color_poses_, eval_depth_poses_; // [N, 4, 4]
  int dataset_system_type_;
  sensor::Sensors sensor_;
  bool preload_;
  float res_scale_;
  int depth_type_ = DepthType::Image; // 0: image; 1: ply; 2: bin; 3: pcd
  torch::Tensor
      T_B_S_; // [4, 4]; extrinsic param, transformation from sensor to body
  torch::Tensor T_S_B_;

  std::vector<std::filesystem::path> raw_color_filelists_, raw_depth_filelists_,
      train_color_filelists_, train_depth_filelists_, eval_color_filelists_,
      eval_depth_filelists_;

  float color_scale_inv_ = 1.0f / 255.0f;
  float depth_scale_inv_ = 1e-3f;

  float max_time_diff_camera_and_pose_, max_time_diff_lidar_and_pose_ = 0.0f;
  int ds_pt_num_;

  torch::Tensor mask = torch::Tensor();

  virtual torch::Tensor get_pose(const int &idx, const int &pose_type) const;

  virtual torch::Tensor get_pose(const torch::Tensor &idx,
                                 const int &pose_type) const;

  virtual torch::Tensor get_image(const int &idx, const int &image_type,
                                  const float &scale = 1.0f) const;
  virtual torch::Tensor get_image(const torch::Tensor &idx,
                                  const int &image_type,
                                  const float &scale = 1.0f) const;
  virtual torch::Tensor get_color_image(const std::string &file_path,
                                        const int &image_type,
                                        const float &scale = 1.0) const;
  virtual torch::Tensor cv_mat_to_tensor(cv::Mat color_mat,
                                         const float &scale) const;

  size_t size(const int &data_type) const {
    switch (data_type) {
    case DataType::RawColor: {
      return raw_color_filelists_.size();
    }
    case DataType::RawDepth: {
      return raw_depth_filelists_.size();
    }
    case DataType::TrainColor: {
      return train_color_filelists_.size();
    }
    case DataType::TrainDepth: {
      return train_depth_filelists_.size();
      ;
    }
    case DataType::EvalColor: {
      return eval_color_filelists_.size();
    }
    case DataType::EvalDepth: {
      return eval_depth_filelists_.size();
    }
    case DataType::TestColor:
    case DataType::TestDepth:
    default:
      throw std::runtime_error("Invalid image type");
    }
  }

  std::filesystem::path get_file(const int &idx, const int &image_type = 0);
  cv::Mat get_image_cv_mat(const int &idx, const int &image_type = 0) const;
  cv::Mat get_image_cv_mat(const std::string &file_path,
                           const int &image_type) const;
  torch::Tensor get_color_image(const int &idx, const int &image_type = 0,
                                const float &scale = 1.0) const;
  torch::Tensor get_depth_image(const int &idx) const;

  virtual std::vector<at::Tensor> get_depth_zdir(const int &idx);
  virtual std::vector<at::Tensor> get_distance_ndir_zdirn(const int &idx);
  virtual at::Tensor get_points(const int &idx);
  virtual std::vector<at::Tensor> get_points_dist_ndir_zdirn(const int &idx);

  virtual void load_data() { throw std::runtime_error("Not implemented"); }

  virtual void load_eval_data() { throw std::runtime_error("Not implemented"); }

  virtual bool load_calib() {
    throw std::runtime_error("Not implemented");

    if (!std::filesystem::exists(calib_path_)) {
      std::cerr << "Calibration file does not exist\n";
      T_B_S_ = torch::eye(4);
    }
    /* else {
      std::ifstream import_file(calib_path_, std::ios::in);
      if (!import_file) {
        std::cerr << "Could not open calibration file: " << calib_path_ <<
    "\n"; return false;
      }

      std::string line;
      while (std::getline(import_file, line)) {
        std::stringstream line_stream(line);

        // Check what the header is. Each line consists of two parts:
        // a header followed by a ':' followed by space-separated data.
        std::string header;
        std::getline(line_stream, header, ':');
        std::string data;
        std::getline(line_stream, data, ':');

        std::vector<float> parsed_floats;
        if (header == "Tr") {
          // Parse the translation matrix.
          if (parseVectorOfFloats(data, &parsed_floats)) {
            Tr =
                torch::from_blob(parsed_floats.data(), {3, 4},
    torch::kFloat32).clone(); Tr = torch::cat({Tr, torch::tensor({{0, 0, 0,
    1}})}, 0);
          }
        }
      }
    } */
    std::cout << "T_B_S:" << "\n" << T_B_S_ << "\n";
    T_S_B_ = T_B_S_.inverse();

    return true;
  }

  virtual std::tuple<torch::Tensor, torch::Tensor,
                     std::vector<std::filesystem::path>>
  load_poses(const std::string &pose_path, const bool &with_head,
             const int &pose_type = 0, bool skip_line = false,
             std::string filter_name = "", bool inverse = false);

  virtual void load_intrinsics();

  virtual void
  align_pose_sensor(std::vector<std::filesystem::path> &out_filelsits,
                    torch::Tensor &sensor_poses,
                    float max_time_diff_sensor_and_pose);

  virtual void load_colors(const std::string &file_extension,
                           const std::string &prefix = "",
                           const bool eval = false, const bool &llff = false);

  virtual void load_depths(const std::string &file_extension,
                           const std::string &prefix = "",
                           const bool eval = false, const bool &llff = false);

  virtual std::string get_gt_mesh_path() { return ""; }

  void post_process(int skip_first_num);
};
} // namespace dataparser