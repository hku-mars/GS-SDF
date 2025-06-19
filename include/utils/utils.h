#pragma once

#include <filesystem>
#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <torch/torch.h>

#include <cuda_runtime_api.h>

#include "ray_utils/ray_utils.h"

#ifdef ENABLE_ROS
#include <pcl_conversions/pcl_conversions.h>

#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Float32.h>
#include <visualization_msgs/MarkerArray.h>
#endif

namespace utils {

std::vector<torch::Tensor> unique(const torch::Tensor &x, const int &dim = 0);

// get cpu data memory usage
double get_cpu_mem_usage();

double get_gpu_mem_usage();

torch::Tensor downsample_point(const torch::Tensor &points, int ds_pt_num);

template <typename RaySamplesT>
RaySamplesT downsample_sample(const RaySamplesT &samples, int ds_pt_num);

DepthSamples downsample(const DepthSamples &samples, float pt_size,
                        float dir_size);
ColorSamples downsample(const ColorSamples &samples, float pt_size,
                        float dir_size);

void extract_ray_depth_from_pcl(const torch::Tensor &pointcloud,
                                torch::Tensor &rays_d, torch::Tensor &depths,
                                float min_range = 0.01);

torch::Tensor pointcloud_to_tensor(PointCloudT &pointcloud);

torch::Tensor vec_eigen_to_tensor(
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>
        &vec_eigen);

std::vector<float> tensor_to_vector(const torch::Tensor &_tensor);

pcl::PointCloud<pcl::PointXYZI>
tensor_to_pointcloud(const torch::Tensor &_tensor);
pcl::PointCloud<pcl::PointXYZRGB>
tensor_to_pointcloud(const torch::Tensor &_xyz, const torch::Tensor &_rgb);

cv::Mat
apply_colormap_to_depth(const cv::Mat &depth, double near = 0.0,
                        double far = 0.0,
                        cv::ColormapTypes colormap = cv::COLORMAP_TURBO);
cv::Mat tensor_to_cv_mat(const torch::Tensor &_image,
                         const float &_depth_scale = 1000.0);

void pointcloud_to_raydepth(PointCloudT &_pointcloud, int _ds_pt_num,
                            torch::Tensor &_rays_d, torch::Tensor &_depths,
                            float min_range = 0.01,
                            torch::Device device = torch::kCPU);

template <typename RaySamplesT>
RaySamplesT sample_batch_pts(const RaySamplesT &_samples, int batch_pt_num = -1,
                             int batch_type = 1, int iter = 0);

DepthSamples sample_surface_pts(const DepthSamples &_samples,
                                int surface_sample_num = 4, float std = 0.1);

DepthSamples sample_free_pts(const DepthSamples &_samples, int sample_num);

void sample_strat_pts(const DepthSamples &_samples, torch::Tensor &sample_pts,
                      torch::Tensor &sample_dirs, torch::Tensor &sample_gts,
                      int strat_sample_num = 4, float std = 0.1,
                      float strat_near_ratio = 0.2,
                      float strat_far_ratio = 0.8);

torch::Tensor cal_nn_dist(const torch::Tensor &_input_pts,
                          const torch::Tensor &_target_pts);

torch::Tensor get_verteices(const torch::Tensor &xyz_index);

torch::Tensor get_width_verteices(const torch::Tensor &xyz_index,
                                  float width = 1.0);

torch::Tensor cal_tri_inter_coef(const torch::Tensor &xyz_weight);

torch::Tensor cal_inter_pair_coef(const torch::Tensor &vertex_sdf,
                                  const torch::Tensor &face_edge_pair_index,
                                  float iso_value = 0.0);

torch::Tensor normalized_quat_to_rotmat(const torch::Tensor &quat);

torch::Tensor quat_to_rot(const torch::Tensor &quat, const bool &xyzw = false);

torch::Tensor rot_to_quat(const torch::Tensor &rotation);

torch::Tensor positional_encode(const torch::Tensor &xyz);

torch::Tensor meshgrid_3d(float x_min, float x_max, float y_min, float y_max,
                          float z_min, float z_max, float resolution,
                          torch::Device &device);

torch::Tensor rotation_6d_to_matrix(const torch::Tensor &rotation6d);

#ifdef ENABLE_ROS
sensor_msgs::PointCloud2 tensor_to_pointcloud_msg(const torch::Tensor &_xyz,
                                                  const torch::Tensor &_rgb);

sensor_msgs::Image tensor_to_img_msg(const torch::Tensor &_image);

visualization_msgs::Marker get_vix_voxel_map(const torch::Tensor &_xyz,
                                             float voxel_size, float r = 1.0,
                                             float g = 1.0, float b = 1.0);
visualization_msgs::Marker get_vis_shift_map(torch::Tensor _pos_W_M,
                                             float _x_min, float _x_max,
                                             float _y_min, float _y_max,
                                             float _z_min, float _z_max);
#endif
} // namespace utils