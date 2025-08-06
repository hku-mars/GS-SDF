#include "data_loader.h"
#include <fstream>
#include <iostream>
#include <memory>

#include "utils/bin_utils/endian.h"
#include "utils/ray_utils/ray_utils.h"

#include "data_loader/data_parsers/colmap_parser.hpp"
#include "data_loader/data_parsers/fastlivo_parser.hpp"
#include "data_loader/data_parsers/kitti_parser.hpp"
#include "data_loader/data_parsers/oxford_spires_parser.hpp"
#include "data_loader/data_parsers/r3live_parser.hpp"
#include "data_loader/data_parsers/replica_parser.hpp"
#include "data_parsers/neuralrgbd_parser.hpp"
#include "pcl/io/pcd_io.h"
#include "utils/ply_utils/ply_utils_pcl.h"
#include "utils/ply_utils/ply_utils_torch.h"

enum DatasetType {
  Replica = 0,
  R3live = 1,
  NeuralRGBD = 2,
  Kitti = 3,
  Fastlivo = 4,
  Spires = 5,
  Colmap = 6,
};

namespace dataloader {
DataLoader::DataLoader(const std::string &dataset_path,
                       const std::filesystem::path &_config_path,
                       const int &_dataset_type, const torch::Device &_device,
                       const bool &_preload, const float &_res_scale,
                       const sensor::Sensors &_sensor, const int &_ds_pt_num,
                       const float &_max_time_diff_camera_and_pose,
                       const float &_max_time_diff_lidar_and_pose)
    : dataset_type_(_dataset_type), device_(_device) {
  switch (dataset_type_) {
  case DatasetType::NeuralRGBD:
    dataparser_ptr_ = std::make_shared<dataparser::NeuralRGBD>(
        dataset_path, device_, _preload, _res_scale, 0, _ds_pt_num);
    break;
  case DatasetType::Replica:
    dataparser_ptr_ = std::make_shared<dataparser::Replica>(
        dataset_path, device_, _preload, _res_scale, _ds_pt_num);
    break;
  case DatasetType::R3live:
    dataparser_ptr_ = std::make_shared<dataparser::R3live>(
        dataset_path, device_, _preload, _res_scale, _ds_pt_num);
    break;
  case DatasetType::Kitti:
    dataparser_ptr_ = std::make_shared<dataparser::Kitti>(
        dataset_path, device_, _preload, _res_scale, _ds_pt_num);
    break;
  case DatasetType::Fastlivo:
    dataparser_ptr_ = std::make_shared<dataparser::Fastlivo>(
        dataset_path, device_, _preload, _res_scale, _sensor, _ds_pt_num);
    break;
  case DatasetType::Spires:
    dataparser_ptr_ = std::make_shared<dataparser::Spires>(
        dataset_path, device_, _preload, _res_scale, _sensor, _ds_pt_num,
        _max_time_diff_camera_and_pose, _max_time_diff_lidar_and_pose);
    break;
  case DatasetType::Colmap:
    dataparser_ptr_ = std::make_shared<dataparser::Colmap>(
        dataset_path, _config_path, device_, _preload, _res_scale, _sensor,
        _ds_pt_num, _max_time_diff_camera_and_pose,
        _max_time_diff_lidar_and_pose);
    break;
  default:
    throw std::runtime_error("Unsupported dataset type");
  }
}

bool DataLoader::get_next_data(int idx, torch::Tensor &_pose,
                               PointCloudT &_points) {
  if (idx >= dataparser_ptr_->raw_depth_filelists_.size()) {
    std::cout << "\nEnd of the data!\n";
    return false;
  }

  std::cout << "\rData idx: " << idx
            << ", Depth file:" << dataparser_ptr_->raw_depth_filelists_[idx];
  _pose = get_pose(idx, dataparser::DataType::RawDepth).to(device_);

  std::string infile =
      dataparser_ptr_->dataset_path_ /
      dataparser_ptr_->get_file(idx, dataparser::DataType::RawDepth);

  if (infile.find(".bin") != std::string::npos) {
    std::ifstream input(infile.c_str(), std::ios::in | std::ios::binary);
    if (!input) {
      std::cerr << "Could not read file: " << infile << "\n";
      return false;
    }

    const size_t kMaxNumberOfPoints = 1e6; // From the Readme of raw files.
    _points.clear();
    _points.reserve(kMaxNumberOfPoints);

    while (input.is_open() && !input.eof()) {
      PointT point;

      input.read((char *)&point.x, 3 * sizeof(float));
      // pcl::PointXYZI
      float intensity;
      input.read((char *)&intensity, sizeof(float));
      // input.read((char *)&point.intensity, sizeof(float));
      _points.push_back(point);
    }
    input.close();
  } else if (infile.find(".ply") != std::string::npos) {
    ply_utils::read_ply_file(infile, _points);
  } else if (infile.find(".pcd") != std::string::npos) {
    pcl::io::loadPCDFile<PointT>(infile, _points);
  }

  return true;
}

bool DataLoader::get_next_data(int idx, torch::Tensor &_pose,
                               DepthSamples &_depth_rays,
                               ColorSamples &_color_rays,
                               const torch::Device &_device) {
  if (idx >= dataparser_ptr_->size(1)) {
    std::cout << "\nEnd of the data!\n";
    return false;
  }

  get_next_data(idx, _pose, _depth_rays, device_);
  torch::Tensor color_pose;
  get_next_data(idx, color_pose, _color_rays, device_);
  return true;
}

bool DataLoader::get_next_data(int idx, torch::Tensor &_pose,
                               DepthSamples &_depth_rays,
                               const torch::Device &_device) {
  if (idx >= dataparser_ptr_->size(1)) {
    std::cout << "\nEnd of the data!\n";
    return false;
  }

  std::cout << "\rData idx: " << idx
            << ", Depth file:" << dataparser_ptr_->raw_depth_filelists_[idx];

  auto points_ndir_dirn = dataparser_ptr_->get_distance_ndir_zdirn(idx);
  DepthSamples depth_rays;
  depth_rays.depth = points_ndir_dirn[0].view({-1, 1}).to(_device);

  depth_rays.direction = points_ndir_dirn[1].view({-1, 3}).to(_device);

  _pose = get_pose(idx, dataparser::DataType::RawDepth).to(_device);
  auto rot = _pose.slice(1, 0, 3);
  auto pos = _pose.slice(1, 3, 4).squeeze();
  //[n,3]
  depth_rays.origin = pos.expand({depth_rays.depth.size(0), 3});
  depth_rays.direction = depth_rays.direction.mm(rot.t());
  depth_rays.xyz = depth_rays.direction * depth_rays.depth + pos;
  _depth_rays = depth_rays;
  return true;
}

bool DataLoader::get_next_data(int idx, torch::Tensor &_pose,
                               ColorSamples &_color_rays,
                               const torch::Device &_device) {
  if (idx >= dataparser_ptr_->size(0)) {
    std::cout << "\nEnd of the data!\n";
    return false;
  }
  // std::cout << "\rData idx: " << idx
  //           << ", Color file:" << dataparser_ptr_->color_filelists_[idx];

  auto rgb = dataparser_ptr_->get_image(idx, dataparser::DataType::RawColor);
  if (rgb.numel() == 0) {
    _color_rays = ColorSamples();
  } else {
    _color_rays.rgb = rgb.view({-1, 3}).to(_device);

    _pose = get_pose(idx, dataparser::DataType::RawColor).to(_device);

    auto camera_ray_results =
        dataparser_ptr_->sensor_.camera.generate_rays(_pose);
    _color_rays.origin = camera_ray_results[0];
    _color_rays.direction = camera_ray_results[1];
  }
  return true;
}

torch::Tensor DataLoader::get_pose(int idx, const int &pose_type) {
  // return pose matrix at idx with shape [3, 4]
  return dataparser_ptr_->get_pose(idx, pose_type).slice(0, 0, 3);
}

int DataLoader::export_image(std::filesystem::path &colmap_sparse_path,
                             std::filesystem::path &colmap_path, int data_type,
                             std::string dirname, bool bin, uint32_t camera_id,
                             bool eval, bool llff, std::string prefix,
                             int prefix_num) {
  if (dataparser_ptr_->size(data_type) == 0) {
    return 0;
  }
  std::ofstream colmap_color_pose_file, test_colmap_color_pose_file;
  if (bin) {
    auto colmap_color_pose_path_ = colmap_sparse_path / (dirname + ".bin");
    colmap_color_pose_file = std::ofstream(colmap_color_pose_path_,
                                           std::ios::trunc | std::ios::binary);

    uint64_t num_reg_images = dataparser_ptr_->size(data_type);
    uint64_t test_num = 0;
    if (llff) {
      test_num = num_reg_images / 8 + 1;
      num_reg_images -= test_num;
    }
    colmap::WriteBinaryLittleEndian<uint64_t>(&colmap_color_pose_file,
                                              num_reg_images);

    auto test_colmap_color_pose_path_ =
        colmap_path / "../test_colmap/sparse/0" / (dirname + ".bin");
    test_colmap_color_pose_file = std::ofstream(
        test_colmap_color_pose_path_, std::ios::app | std::ios::binary);
    colmap::WriteBinaryLittleEndian<uint64_t>(&test_colmap_color_pose_file,
                                              test_num);
  } else {
    auto colmap_color_pose_path_ = colmap_sparse_path / (dirname + ".txt");
    colmap_color_pose_file =
        std::ofstream(colmap_color_pose_path_, std::ios::app);

    auto test_colmap_color_pose_path_ =
        colmap_path / "../test_colmap/sparse/0" / (dirname + ".txt");
    test_colmap_color_pose_file = std::ofstream(test_colmap_color_pose_path_);
  }

  auto colmap_color_file_path = colmap_path / dirname;
  std::filesystem::create_directory(colmap_color_file_path);
  auto test_colmap_color_file_path = colmap_path / "../test_colmap" / dirname;
  std::filesystem::create_directory(test_colmap_color_file_path);

  for (uint32_t i = 0; i < dataparser_ptr_->size(data_type); ++i) {
    auto file_path = dataparser_ptr_->get_file(i, data_type);
    std::cout << '\r' << i << " " << file_path;
    // extrace number in filename and append with 5 digital format
    auto num = file_path.filename().string();
    num = num.substr(num.find_first_of("0123456789"));
    num = num.substr(0, num.find_last_of('.'));
    char buffer[256];
    std::sprintf(buffer, "%s%05d%s", prefix.c_str(), std::stoi(num),
                 file_path.extension().c_str());
    std::string filename = std::string(buffer);

    // https://colmap.github.io/format.html#images-txt
    auto pose = get_pose(i, data_type);
    auto rot_world_to_camera = pose.slice(1, 0, 3).t();
    auto pos_world_to_camera = -rot_world_to_camera.mm(pose.slice(1, 3, 4));
    auto quat = utils::rot_to_quat(rot_world_to_camera).to(torch::kDouble);

    std::ofstream *tmp_pose_file;
    std::filesystem::path *tmp_file_path;
    if (llff && (i % 8 == 0)) {
      tmp_pose_file = &test_colmap_color_pose_file;
      tmp_file_path = &test_colmap_color_file_path;
    } else {
      tmp_pose_file = &colmap_color_pose_file;
      tmp_file_path = &colmap_color_file_path;
    }
    std::filesystem::copy_file(
        file_path, (*tmp_file_path) / filename,
        std::filesystem::copy_options::overwrite_existing);

    if (bin) {
      colmap::WriteBinaryLittleEndian<uint32_t>(tmp_pose_file, prefix_num + i);

      colmap::WriteBinaryLittleEndian<double>(tmp_pose_file,
                                              quat[0].item<double>());
      colmap::WriteBinaryLittleEndian<double>(tmp_pose_file,
                                              quat[1].item<double>());
      colmap::WriteBinaryLittleEndian<double>(tmp_pose_file,
                                              quat[2].item<double>());
      colmap::WriteBinaryLittleEndian<double>(tmp_pose_file,
                                              quat[3].item<double>());

      colmap::WriteBinaryLittleEndian<double>(
          tmp_pose_file, pos_world_to_camera[0][0].item<double>());
      colmap::WriteBinaryLittleEndian<double>(
          tmp_pose_file, pos_world_to_camera[1][0].item<double>());
      colmap::WriteBinaryLittleEndian<double>(
          tmp_pose_file, pos_world_to_camera[2][0].item<double>());

      colmap::WriteBinaryLittleEndian<uint32_t>(tmp_pose_file, camera_id);
      const std::string name = filename + '\0';
      (*tmp_pose_file).write(name.c_str(), name.size());

      uint64_t num_points2D = 0;
      colmap::WriteBinaryLittleEndian<uint64_t>(tmp_pose_file, num_points2D);
    } else {
      // #   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
      // #   POINTS2D[] as (X, Y, POINT3D_ID);
      (*tmp_pose_file) << std::to_string(prefix_num + i) << " ";
      (*tmp_pose_file) << quat[0].item<float>() << " " << quat[1].item<float>()
                       << " " << quat[2].item<float>() << " "
                       << quat[3].item<float>() << " ";
      (*tmp_pose_file) << pos_world_to_camera[0][0].item<float>() << " "
                       << pos_world_to_camera[1][0].item<float>() << " "
                       << pos_world_to_camera[2][0].item<float>() << " ";
      (*tmp_pose_file) << camera_id << " " << filename << "\n";
      (*tmp_pose_file) << "\n";
    }
  }
  colmap_color_pose_file.close();
  return dataparser_ptr_->size(data_type);
}

void DataLoader::export_as_colmap_format(bool bin, bool llff) {
  /* 3DGS: bin=true, llff=false;
     Others: bin=true, llff=true; to separate llff eval standard in test dir */
  std::cout << "Exporting as colmap format\n";
  // export for colmap
  auto colmap_path = dataparser_ptr_->dataset_path_ / "colmap";
  auto test_colmap_path = dataparser_ptr_->dataset_path_ / "test_colmap";
  if (std::filesystem::exists(colmap_path)) {
    // std::filesystem::remove_all(colmap_path_sparse_path);
    std::cout << "Colmap format already exists\n";
    return;
  }
  auto colmap_sparse_path = colmap_path / "sparse/0";
  std::filesystem::create_directories(colmap_sparse_path);
  auto test_colmap_sparse_path = test_colmap_path / "sparse/0";
  std::filesystem::create_directories(test_colmap_sparse_path);

  std::ofstream colmap_camera_file;
  uint32_t camera_id = 1;
  if (bin) {
    auto colmap_camera_path = colmap_sparse_path / "cameras.bin";
    colmap_camera_file =
        std::ofstream(colmap_camera_path, std::ios::trunc | std::ios::binary);
    uint64_t num_cameras = 1;
    colmap::WriteBinaryLittleEndian<uint64_t>(&colmap_camera_file, num_cameras);
    colmap::WriteBinaryLittleEndian<uint32_t>(&colmap_camera_file, camera_id);

    int model_id = 1; // PINHOLE
    colmap::WriteBinaryLittleEndian<int>(&colmap_camera_file,
                                         static_cast<int>(model_id));
    colmap::WriteBinaryLittleEndian<uint64_t>(
        &colmap_camera_file, (uint64_t)dataparser_ptr_->sensor_.camera.width);
    colmap::WriteBinaryLittleEndian<uint64_t>(
        &colmap_camera_file, (uint64_t)dataparser_ptr_->sensor_.camera.height);
    colmap::WriteBinaryLittleEndian<double>(
        &colmap_camera_file, (double)dataparser_ptr_->sensor_.camera.fx);
    colmap::WriteBinaryLittleEndian<double>(
        &colmap_camera_file, (double)dataparser_ptr_->sensor_.camera.fy);
    colmap::WriteBinaryLittleEndian<double>(
        &colmap_camera_file, (double)dataparser_ptr_->sensor_.camera.cx);
    colmap::WriteBinaryLittleEndian<double>(
        &colmap_camera_file, (double)dataparser_ptr_->sensor_.camera.cy);
    colmap_camera_file.close();

    // copy to test colmap
    std::filesystem::copy_file(
        colmap_camera_path, test_colmap_sparse_path / "cameras.bin",
        std::filesystem::copy_options::overwrite_existing);
  } else {
    auto colmap_camera_path = colmap_sparse_path / "cameras.txt";
    colmap_camera_file = std::ofstream(colmap_camera_path);
    // CAMERA_ID MODEL WIDTH HEIGHT fx fy cx cy k1 k2 p1 p2
    colmap_camera_file << camera_id << " PINHOLE "
                       << dataparser_ptr_->sensor_.camera.width << " "
                       << dataparser_ptr_->sensor_.camera.height << " "
                       << dataparser_ptr_->sensor_.camera.fx << " "
                       << dataparser_ptr_->sensor_.camera.fy << " "
                       << dataparser_ptr_->sensor_.camera.cx << " "
                       << dataparser_ptr_->sensor_.camera.cy << " "
                       << dataparser_ptr_->sensor_.camera.d0 << " "
                       << dataparser_ptr_->sensor_.camera.d1 << " "
                       << dataparser_ptr_->sensor_.camera.d2 << " "
                       << dataparser_ptr_->sensor_.camera.d3 << "\n";
    colmap_camera_file.close();

    // copy to test colmap
    std::filesystem::copy_file(
        colmap_camera_path, test_colmap_sparse_path / "cameras.txt",
        std::filesystem::copy_options::overwrite_existing);
  }

  export_image(colmap_sparse_path, colmap_path, dataparser::DataType::RawColor,
               "images", bin, camera_id, false, llff);

  export_image(colmap_sparse_path, colmap_path, dataparser::DataType::RawDepth,
               "depths", bin, camera_id, false, llff);
  if (!dataparser_ptr_->eval_color_filelists_.empty()) {
    export_image(colmap_sparse_path, colmap_path,
                 dataparser::DataType::EvalColor, "eval_images", bin, camera_id,
                 true, false);
    export_image(colmap_sparse_path, colmap_path,
                 dataparser::DataType::EvalDepth, "eval_depths", bin, camera_id,
                 true, false);
  }

  // constraint the number of points to avoid OOM in 3DGS
  auto train_depth_xyz = dataparser_ptr_->train_depth_pack_.xyz.view({-1, 3});
  float downsample_pt_num = 1e7f;
  int sample_step = train_depth_xyz.size(0) / downsample_pt_num;
  sample_step = std::max(sample_step, 1);
  auto points = train_depth_xyz.slice(0, 0, -1, sample_step).to(torch::kDouble);

  if (bin) {
    auto colmap_points_path = colmap_sparse_path / "points3D.bin";
    std::ofstream colmap_points_file(colmap_points_path, std::ios::binary);

    uint64_t num_points = points.size(0);
    colmap::WriteBinaryLittleEndian<uint64_t>(&colmap_points_file, num_points);

    auto points_acc = points.accessor<double, 2>();
    for (uint64_t i = 0; i < num_points; ++i) {
      colmap::WriteBinaryLittleEndian<uint64_t>(&colmap_points_file, i);
      colmap::WriteBinaryLittleEndian<double>(&colmap_points_file,
                                              points_acc[i][0]);
      colmap::WriteBinaryLittleEndian<double>(&colmap_points_file,
                                              points_acc[i][1]);
      colmap::WriteBinaryLittleEndian<double>(&colmap_points_file,
                                              points_acc[i][2]);
      colmap::WriteBinaryLittleEndian<uint8_t>(&colmap_points_file, 255);
      colmap::WriteBinaryLittleEndian<uint8_t>(&colmap_points_file, 255);
      colmap::WriteBinaryLittleEndian<uint8_t>(&colmap_points_file, 255);
      colmap::WriteBinaryLittleEndian<double>(&colmap_points_file, 0.0);

      uint64_t track_length = 0;
      colmap::WriteBinaryLittleEndian<uint64_t>(&colmap_points_file,
                                                track_length);
    }
    colmap_points_file.close();
  } else {
    std::filesystem::path colmap_points_path =
        colmap_sparse_path / "points3D.ply";
    auto colmap_rgb = torch::zeros_like(points);
    auto colmap_normal = torch::zeros_like(points);
    ply_utils::export_to_ply(colmap_points_path, points, colmap_rgb, {}, {}, {},
                             colmap_normal);
  }
  std::cout << "Exported as 3DGS format at: " << colmap_path << "\n";
}

void DataLoader::export_as_colmap_format_for_nerfstudio(bool bin) {
  /* 3DGS: bin=true, llff=false;
     Others: bin=true, llff=true; to separate llff eval standard in test dir */
  std::cout << "Exporting as colmap format\n";
  // export for colmap
  auto colmap_path = dataparser_ptr_->dataset_path_ / "nerfstudio";
  if (std::filesystem::exists(colmap_path)) {
    // std::filesystem::remove_all(colmap_path_sparse_path);
    std::cout << "Colmap format already exists\n";
    return;
  }
  // remove all existing files
  std::filesystem::remove_all(colmap_path);
  auto colmap_sparse_path = colmap_path / "colmap/sparse/0";
  std::filesystem::create_directories(colmap_sparse_path);

  std::ofstream colmap_camera_file;
  uint32_t camera_id = 1;
  auto colmap_camera_path = colmap_sparse_path / "cameras.txt";
  colmap_camera_file = std::ofstream(colmap_camera_path);
  // CAMERA_ID MODEL WIDTH HEIGHT fx fy cx cy k1 k2 p1 p2
  colmap_camera_file << camera_id << " PINHOLE "
                     << dataparser_ptr_->sensor_.camera.width << " "
                     << dataparser_ptr_->sensor_.camera.height << " "
                     << dataparser_ptr_->sensor_.camera.fx << " "
                     << dataparser_ptr_->sensor_.camera.fy << " "
                     << dataparser_ptr_->sensor_.camera.cx << " "
                     << dataparser_ptr_->sensor_.camera.cy << " "
                     << dataparser_ptr_->sensor_.camera.d0 << " "
                     << dataparser_ptr_->sensor_.camera.d1 << " "
                     << dataparser_ptr_->sensor_.camera.d2 << " "
                     << dataparser_ptr_->sensor_.camera.d3 << "\n";
  colmap_camera_file.close();

  auto color_prefix_num =
      export_image(colmap_sparse_path, colmap_path, 0, "images", bin, camera_id,
                   false, false, "train_");
  auto depth_prefix_num =
      export_image(colmap_sparse_path, colmap_path, 1, "depths", bin, camera_id,
                   false, false, "train_");
  if (!dataparser_ptr_->eval_color_filelists_.empty()) {
    export_image(colmap_sparse_path, colmap_path, 0, "images", bin, camera_id,
                 true, false, "eval_", color_prefix_num);
  }

  // constraint the number of points to avoid OOM in 3DGS

  auto train_depth_xyz = dataparser_ptr_->train_depth_pack_.xyz.view({-1, 3});
  float downsample_pt_num = 1e7f;
  int sample_step = train_depth_xyz.size(0) / downsample_pt_num;
  sample_step = std::max(sample_step, 1);
  auto points = train_depth_xyz.slice(0, 0, -1, sample_step).to(torch::kDouble);

  auto colmap_points_path = colmap_sparse_path / "points3D.bin";
  std::ofstream colmap_points_file(colmap_points_path, std::ios::binary);

  uint64_t num_points = points.size(0);
  colmap::WriteBinaryLittleEndian<uint64_t>(&colmap_points_file, num_points);

  auto points_acc = points.accessor<double, 2>();
  for (uint64_t i = 0; i < num_points; ++i) {
    colmap::WriteBinaryLittleEndian<uint64_t>(&colmap_points_file, i);
    colmap::WriteBinaryLittleEndian<double>(&colmap_points_file,
                                            points_acc[i][0]);
    colmap::WriteBinaryLittleEndian<double>(&colmap_points_file,
                                            points_acc[i][1]);
    colmap::WriteBinaryLittleEndian<double>(&colmap_points_file,
                                            points_acc[i][2]);
    colmap::WriteBinaryLittleEndian<uint8_t>(&colmap_points_file, 255);
    colmap::WriteBinaryLittleEndian<uint8_t>(&colmap_points_file, 255);
    colmap::WriteBinaryLittleEndian<uint8_t>(&colmap_points_file, 255);
    colmap::WriteBinaryLittleEndian<double>(&colmap_points_file, 0.0);

    uint64_t track_length = 0;
    colmap::WriteBinaryLittleEndian<uint64_t>(&colmap_points_file,
                                              track_length);
  }
  colmap_points_file.close();

  std::cout << "Exported as 3DGS format at: " << colmap_path << "\n";
}
} // namespace dataloader