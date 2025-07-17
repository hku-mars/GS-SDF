#include "utils.h"
#include "kaolin_wisp_cpp/spc_ops/spc_ops.h"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <torch/torch.h>

#include <cuda_runtime_api.h>

using namespace std;

namespace utils {

std::vector<torch::Tensor> unique(const torch::Tensor &x, const int &dim) {
  // https://github.com/pytorch/pytorch/issues/36748#issuecomment-1474368922
  // https://github.com/pytorch/pytorch/issues/70920
  // TODO: return index not right
  auto unique_resutls = torch::unique_dim(x, dim, true, true, true);
  auto unique = std::get<0>(unique_resutls);
  auto inverse = std::get<1>(unique_resutls);
  auto counts = std::get<2>(unique_resutls);
  auto inv_sorted = inverse.argsort(true);
  auto tot_counts =
      torch::cat({counts.new_zeros(1), counts.cumsum(0)}).slice(0, 0, -1);
  auto index = inv_sorted.index_select(0, tot_counts);
  return {unique, inverse, counts, index};
}

// get cpu data memory usage
double get_cpu_mem_usage() {
  FILE *file = fopen("/proc/self/status", "r");
  int result = -1;
  char line[128];

  while (fgets(line, 128, file) != nullptr) {
    if (strncmp(line, "VmRSS:", 6) == 0) {
      int len = strlen(line);

      const char *p = line;
      for (; std::isdigit(*p) == false; ++p) {
      }

      line[len - 3] = 0;
      result = atoi(p);

      break;
    }
  }
  fclose(file);

  std::cout << "\nNow used CPU memory " << result / 1024.0 / 1024.0 << "  GB\n";

  return (result / 1024.0 / 1024.0);
}

double get_gpu_mem_usage() {
  size_t free_byte;
  size_t total_byte;

  cudaError_t cuda_status = cudaMemGetInfo(&free_byte, &total_byte);

  if (cudaSuccess != cuda_status) {
    printf("Error: cudaMemGetInfo fails, %s \n",
           cudaGetErrorString(cuda_status));
    exit(1);
  }

  auto free_db = (double)free_byte;
  auto total_db = (double)total_byte;
  double used_db = (total_db - free_db) / 1024.0 / 1024.0 / 1024.0;
  std::cout << "Now used GPU memory " << used_db << "  GB\n";
  return used_db;
}

torch::Tensor downsample_point(const torch::Tensor &points, int ds_pt_num) {
  if (points.size(0) > ds_pt_num) {
    int step = points.size(0) / ds_pt_num + 1.0;
    return points.slice(0, 0, -1, step);
  } else {
    return points;
  }
}

template RaySamples downsample_sample<RaySamples>(const RaySamples &samples,
                                                  int ds_pt_num);
template DepthSamples
downsample_sample<DepthSamples>(const DepthSamples &samples, int ds_pt_num);
template ColorSamples
downsample_sample<ColorSamples>(const ColorSamples &samples, int ds_pt_num);
template <typename RaySamplesT>
RaySamplesT downsample_sample(const RaySamplesT &samples, int ds_pt_num) {
  if (samples.size() > ds_pt_num) {
    // int step = samples.size() / ds_pt_num + 1.0;
    // return samples.slice(0, 0, -1, step);
    auto sample_idx =
        torch::randint(0, samples.size(), {ds_pt_num}, torch::kLong)
            .to(samples.device());
    return samples.index_select(0, sample_idx);
  } else {
    return samples;
  }
}

ColorSamples downsample(const ColorSamples &samples, float pt_size,
                        float dir_size) {
  assert(pt_size > 0 && dir_size > 0);

  auto voxelized_origin = samples.origin / pt_size;
  auto voxelized_dir = samples.direction / dir_size;
  auto cat_voxelized_origin_dir =
      torch::cat({voxelized_origin, voxelized_dir}, 1).to(torch::kLong);

  auto unique_index = utils::unique(cat_voxelized_origin_dir, 0)[3];

  return samples.index_select(0, unique_index);
}

void extract_ray_depth_from_pcl(const torch::Tensor &pointcloud,
                                torch::Tensor &rays_d, torch::Tensor &depths,
                                float min_range) {
  depths = pointcloud.norm(2, 1, true);
  rays_d = pointcloud.div(depths);
}

/**
 * @description: PointXYZ: 4; PointXYZRGB(A): 8
 * @param {PointCloudT} &pointcloud
 * @return {*}
 */
torch::Tensor pointcloud_to_tensor(PointCloudT &pointcloud) {
  return torch::from_blob(pointcloud.points.data(),
                          {(long)pointcloud.points.size(), 4}, torch::kFloat)
      .clone()
      .slice(1, 0, 3);
}

torch::Tensor vec_eigen_to_tensor(
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>
        &vec_eigen) {
  return torch::from_blob(vec_eigen.data(), {(long)vec_eigen.size(), 3},
                          torch::kDouble)
      .clone()
      .to(torch::kFloat);
}

std::vector<float> tensor_to_vector(const torch::Tensor &_tensor) {
  auto tensor_cpu = _tensor.cpu().contiguous();
  return std::vector<float>(tensor_cpu.data_ptr<float>(),
                            tensor_cpu.data_ptr<float>() + tensor_cpu.numel());
}

pcl::PointCloud<pcl::PointXYZI>
tensor_to_pointcloud(const torch::Tensor &_tensor) {
  pcl::PointCloud<pcl::PointXYZI> pointcloud;
  pointcloud.points.resize(_tensor.size(0));
  pointcloud.width = _tensor.size(0);
  pointcloud.height = 1;

  torch::Tensor tensor_cat;
  auto device = _tensor.device();
  if (_tensor.size(1) == 3) {
    tensor_cat =
        torch::cat({_tensor, torch::empty({_tensor.size(0), 5}, device)}, 1)
            .cpu()
            .contiguous();
  } else if (_tensor.size(1) == 4) {
    tensor_cat = torch::cat({_tensor.slice(1, 0, 3),
                             torch::empty({_tensor.size(0), 1}, device),
                             _tensor.slice(1, 3, 4),
                             torch::empty({_tensor.size(0), 3}, device)},
                            1)
                     .cpu()
                     .contiguous();
  } else {
    std::cout << _tensor.sizes() << std::endl;
    throw std::runtime_error("Not implemented");
  }
  memcpy(pointcloud.points.data(), tensor_cat.data_ptr<float>(),
         tensor_cat.numel() * sizeof(float));
  return pointcloud;
}

pcl::PointCloud<pcl::PointXYZ>
tensor_to_pointcloudXYZ(const torch::Tensor &_tensor) {
  pcl::PointCloud<pcl::PointXYZ> pointcloud;
  pointcloud.points.resize(_tensor.size(0));
  pointcloud.width = _tensor.size(0);
  pointcloud.height = 1;

  auto device = _tensor.device();
  auto tensor_cat =
      torch::cat({_tensor, torch::empty({_tensor.size(0), 1}, device)}, 1);
  auto tensor_cat_cpu = tensor_cat.cpu();

  memcpy(pointcloud.points.data(), tensor_cat_cpu.data_ptr<float>(),
         tensor_cat_cpu.numel() * sizeof(float));
  return pointcloud;
}

torch::Tensor encode_rgb_to_single_float(const torch::Tensor &_rgb) {
  /*
  encode rgb
  * \code
  * // pack r/g/b into rgb
  * std::uint8_t r = 255, g = 0, b = 0;    // Example: Red color
  * std::uint32_t rgb = ((std::uint32_t)r << 16 | (std::uint32_t)g << 8 |
  (std::uint32_t)b);
  * p.rgb = *reinterpret_cast<float*>(&rgb);
  * \endcode
  */
  auto rgb_uint = (_rgb * 255).to(torch::kUInt8);
  auto r = rgb_uint.select(1, 0);
  auto g = rgb_uint.select(1, 1);
  auto b = rgb_uint.select(1, 2);
  torch::Tensor a;
  if (_rgb.size(1) == 3) {
    a = (torch::full({r.size(0)}, 255, r.options()));
  } else {
    a = rgb_uint.select(1, 3);
  }

  auto bgra = torch::stack({b, g, r, a}, 1).contiguous();
  // bitwise stitch
  auto bgra_float = torch::empty({r.size(0), 1}, r.options()).to(torch::kFloat);

  cudaMemcpy(bgra_float.data_ptr(), bgra.data_ptr(),
             bgra.numel() * sizeof(uint8_t), cudaMemcpyDeviceToDevice);
  return bgra_float;
}

pcl::PointCloud<pcl::PointXYZRGB>
tensor_to_pointcloud(const torch::Tensor &_xyz, const torch::Tensor &_rgb) {
  pcl::PointCloud<pcl::PointXYZRGB> pointcloud;
  pointcloud.points.resize(_xyz.size(0));
  pointcloud.width = _xyz.size(0);
  pointcloud.height = 1;

  auto options = _xyz.options();
  auto tensor_cat = torch::cat({_xyz, torch::empty({_xyz.size(0), 1}, options),
                                encode_rgb_to_single_float(_rgb),
                                torch::empty({_xyz.size(0), 3}, options)},
                               1);

  memcpy(pointcloud.points.data(), tensor_cat.cpu().data_ptr<float>(),
         tensor_cat.numel() * sizeof(float));
  return pointcloud;
}

cv::Mat apply_colormap_to_depth(const cv::Mat &depth, double near, double far,
                                cv::ColormapTypes colormap) {
  cv::Mat depth_colormap;
  depth.convertTo(depth_colormap, CV_32FC1);
  auto mask = depth_colormap > 0;
  if ((far - near) < 1e-6) {
    cv::minMaxLoc(depth_colormap, &near, &far, nullptr, nullptr, mask);
  }
  depth_colormap = (depth_colormap - near) / (far - near + 1e-10);
  depth_colormap.setTo(1.f, ~mask);
  depth_colormap.convertTo(depth_colormap, CV_8UC1, 255);
  cv::applyColorMap(depth_colormap, depth_colormap, colormap);
  return depth_colormap;
}

cv::Mat tensor_to_cv_mat(const torch::Tensor &_image,
                         const float &_depth_scale) {
  if (_image.size(-1) == 3) {
    auto img_255 =
        (_image.clamp(0.f, 1.f) * 255).to(torch::kUInt8).cpu().contiguous();
    cv::Mat img(img_255.size(0), img_255.size(1), CV_8UC3);
    memcpy(img.data, img_255.data_ptr(), img_255.numel() * sizeof(uint8_t));
    cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
    return img;
  } else if (_image.size(-1) == 1) {
    auto image = _image.cpu().contiguous();
    cv::Mat depth_img(image.size(0), image.size(1), CV_32FC1);
    memcpy(depth_img.data, image.data_ptr(), image.numel() * sizeof(float));

    depth_img.convertTo(depth_img, CV_16UC1, _depth_scale);
    return depth_img;
  }
  throw std::runtime_error("Not implemented");
}

void pointcloud_to_raydepth(PointCloudT &_pointcloud, int _ds_pt_num,
                            torch::Tensor &_rays_d, torch::Tensor &_depths,
                            float min_range, torch::Device device) {

  // [n,3]
  auto xyz = pointcloud_to_tensor(_pointcloud).to(device);

  // [n,3],[n,3],[n,1]
  extract_ray_depth_from_pcl(xyz, _rays_d, _depths, min_range);
  _rays_d = downsample_point(_rays_d, _ds_pt_num);
  _depths = downsample_point(_depths, _ds_pt_num);
}

template RaySamples sample_batch_pts(const RaySamples &_samples,
                                     int batch_pt_num, int batch_type,
                                     int iter);
template DepthSamples sample_batch_pts(const DepthSamples &_samples,
                                       int batch_pt_num, int batch_type,
                                       int iter);
template ColorSamples sample_batch_pts(const ColorSamples &_samples,
                                       int batch_pt_num, int batch_type,
                                       int iter);
template <typename RaySamplesT>
RaySamplesT sample_batch_pts(const RaySamplesT &_samples, int batch_pt_num,
                             int batch_type, int iter) {
  RaySamplesT batch_samples;
  if (batch_pt_num > 0 && _samples.size() > batch_pt_num) {
    if (batch_type == 0) {
      auto sample_idx =
          torch::randint(0, _samples.size(), {batch_pt_num}, torch::kLong)
              .to(_samples.device());
      batch_samples = _samples.index_select(0, sample_idx);
    } else if (batch_type == 1) {
      int start = (batch_pt_num * iter) % _samples.size();
      int end = start + batch_pt_num;
      batch_samples = _samples.slice(0, start, end);
      if (end > _samples.size()) {
        auto tmp_end = end - _samples.size();
        batch_samples = batch_samples.cat(_samples.slice(0, 0, tmp_end));
      }
    } else if (batch_type == 2) {
      int step = _samples.size() / batch_pt_num + 1.0;
      int rand = torch::randint(0, step, {1}, torch::kLong).item<int>();
      batch_samples = _samples.slice(0, rand, -1, step);
    }
  } else {
    batch_samples = _samples;
  }
  return batch_samples;
}

DepthSamples sample_surface_pts(const DepthSamples &_samples,
                                int surface_sample_num, float std) {
  DepthSamples surface_samples;
  // sample points along rays_d within voxel
  /// [n,k,1]
  surface_samples.ray_sdf =
      torch::randn({_samples.size(), surface_sample_num, 1},
                   _samples.device()) *
      std;
  /// [n,k,3]->[n*k,3]
  surface_samples.origin = _samples.origin.unsqueeze(1)
                               .repeat({1, surface_sample_num, 1})
                               .view({-1, 3});
  surface_samples.xyz =
      (_samples.xyz.unsqueeze(1) -
       _samples.direction.unsqueeze(1) * surface_samples.ray_sdf)
          .view({-1, 3});
  surface_samples.direction = _samples.direction.unsqueeze(1)
                                  .repeat({1, surface_sample_num, 1})
                                  .view({-1, 3});
  surface_samples.ridx =
      _samples.ridx.unsqueeze(1).repeat({1, surface_sample_num}).view({-1});
  surface_samples.depth = _samples.depth.unsqueeze(1)
                              .repeat({1, surface_sample_num, 1})
                              .view({-1, 1});

  surface_samples.ray_sdf = surface_samples.ray_sdf.view({-1, 1});
  return surface_samples;
}

DepthSamples sample_free_pts(const DepthSamples &_samples, int sample_num) {
  // # Sample points along 1D line
  // # depth ~ (NUM_RAYS, NUM_SAMPLES)
  // [N, K]
  torch::Tensor steps = torch::arange(sample_num, _samples.device())
                            .unsqueeze(0)
                            .to(torch::kFloat32)
                            .repeat({_samples.size(0), 1});
  steps += torch::rand_like(steps);
  static float inv_sample_num = 1.0f / sample_num;
  steps *= inv_sample_num;

  // //  non-linearly map depth to distribute more points near 1.0
  // static float beta = 0.5f; // beta < 1 shifts samples towards 1
  // steps = torch::pow(steps, beta);

  // [N * K]
  DepthSamples free_samples = _samples.index_select(
      0, _samples.ridx.unsqueeze(1).repeat({1, sample_num}).view({-1}));

  // [N * K, 1]
  auto depth_free_samples = free_samples.depth * steps.view({-1, 1});
  free_samples.ray_sdf = free_samples.depth - depth_free_samples;
  free_samples.depth = depth_free_samples;
  free_samples.xyz =
      free_samples.origin + free_samples.direction * free_samples.depth;
  return free_samples;
}

void sample_strat_pts(const DepthSamples &_samples, torch::Tensor &sample_pts,
                      torch::Tensor &sample_dirs, torch::Tensor &sample_gts,
                      int strat_sample_num, float std, float strat_near_ratio,
                      float strat_far_ratio) {

  auto device = _samples.device();
  // sample points along rays_d within voxel
  /// [n,k,1]
  auto linspace = torch::linspace(1 - strat_far_ratio, 1 - strat_near_ratio,
                                  strat_sample_num, device)
                      .view({1, strat_sample_num});
  sample_gts = linspace.expand({_samples.size(), strat_sample_num});
  auto strat_sample_rand = std * torch::randn({1, strat_sample_num, 1}, device);
  sample_gts = (_samples.depth * sample_gts).view({-1, strat_sample_num, 1}) +
               strat_sample_rand;
  // [n,k,3]->[n*k,3]
  sample_pts = (_samples.xyz.view({-1, 1, 3}) -
                _samples.direction.unsqueeze(1) * sample_gts);
  sample_dirs =
      _samples.direction.unsqueeze(1).repeat({1, strat_sample_num, 1});
}

torch::Tensor cal_nn_dist(const torch::Tensor &_query_pts,
                          const torch::Tensor &_ref_pts) {
  // static auto p_t_bf_knn = llog::CreateTimer("bf_knn");
  // p_t_bf_knn->tic();
  auto pts_dim = _ref_pts.size(1);
  auto dist =
      (_ref_pts.view({1, -1, pts_dim}) - _query_pts.view({-1, 1, pts_dim}))
          .norm(2, 2);
  auto min_dist = std::get<0>(dist.min(1));
  // p_t_bf_knn->toc_sum();
  return get<0>(dist.min(1));

  /* auto cloud_ref = tensor_to_pointcloudXYZ(_ref_pts);
  auto cloud_query = tensor_to_pointcloudXYZ(_query_pts);
  testCUDA(cloud_ref.makeShared(), cloud_query.makeShared()); */

  /* static auto p_t_cuda_knn = llog::CreateTimer("cuda_knn");
  p_t_cuda_knn->tic();
  auto query_nb = _query_pts.size(0);
  auto ref_nb = _ref_pts.size(0);
  torch::Tensor knn_dist = torch::full({query_nb}, 0.0, torch::kFloat);
  torch::Tensor knn_index = torch::full({query_nb}, -1, torch::kInt);
  // torch::Tensor knn_dist =
  //     torch::full({query_nb}, 0.0,
  //     _query_pts.options().dtype(torch::kFloat));
  // torch::Tensor knn_index =
  //     torch::full({query_nb}, -1, _query_pts.options().dtype(torch::kInt));

  // knn_CUDA requires row-major order
  auto rpts_cpu = _ref_pts.t().contiguous().cpu();
  auto qpts_cpu = _query_pts.t().contiguous().cpu();

  knn_cublas(rpts_cpu.data_ptr<float>(), ref_nb, qpts_cpu.data_ptr<float>(),
             query_nb, _query_pts.size(1), 1, knn_dist.data_ptr<float>(),
             knn_index.data_ptr<int>());
  auto knn_dist_cuda = knn_dist.cuda();
  p_t_cuda_knn->toc_sum(); */

  /* static auto p_t_cuda_knn = llog::CreateTimer("cuda_knn");
  p_t_cuda_knn->tic();
  auto query_nb = _query_pts.size(0);
  auto ref_nb = _ref_pts.size(0);
  torch::Tensor knn_dist =
      torch::full({query_nb}, 0.0, _query_pts.options().dtype(torch::kFloat));
  torch::Tensor knn_index =
      torch::full({query_nb}, -1, _query_pts.options().dtype(torch::kInt));

  // knn_CUDA requires row-major order
  auto rpts = _ref_pts.t().contiguous();
  auto qpts = _query_pts.t().contiguous();

  knn_cublas_cuda(rpts.data_ptr<float>(), ref_nb, qpts.data_ptr<float>(),
                  query_nb, _query_pts.size(1), 1, knn_dist.data_ptr<float>(),
                  knn_index.data_ptr<int>());
  p_t_cuda_knn->toc_sum(); */

  // return knn_dist;
}

torch::Tensor get_verteices(const torch::Tensor &xyz_index) {
  /**
   * @description:
   * @return [n,8,3]
   */
  auto device = xyz_index.device();
  auto tensor_option =
      torch::TensorOptions().dtype(torch::kLong).device(device);
  return torch::stack({xyz_index,
                       xyz_index + torch::tensor({{0, 0, 1}}, tensor_option),
                       xyz_index + torch::tensor({{0, 1, 0}}, tensor_option),
                       xyz_index + torch::tensor({{0, 1, 1}}, tensor_option),
                       xyz_index + torch::tensor({{1, 0, 0}}, tensor_option),
                       xyz_index + torch::tensor({{1, 0, 1}}, tensor_option),
                       xyz_index + torch::tensor({{1, 1, 0}}, tensor_option),
                       xyz_index + torch::tensor({{1, 1, 1}}, tensor_option)},
                      1);
}

torch::Tensor get_width_verteices(const torch::Tensor &xyz_index, float width) {
  auto device = xyz_index.device();
  auto xyz_001 = xyz_index + width * torch::tensor({{0., 0., 1.}}, device); // 1
  auto xyz_010 = xyz_index + width * torch::tensor({{0., 1., 0.}}, device); // 2
  auto xyz_011 = xyz_index + width * torch::tensor({{0., 1., 1.}}, device); // 3
  auto xyz_100 = xyz_index + width * torch::tensor({{1., 0., 0.}}, device); // 4
  auto xyz_101 = xyz_index + width * torch::tensor({{1., 0., 1.}}, device); // 5
  auto xyz_110 = xyz_index + width * torch::tensor({{1., 1., 0.}}, device); // 6
  auto xyz_111 = xyz_index + width * torch::tensor({{1., 1., 1.}}, device); // 7
  return torch::stack({xyz_index, xyz_001, xyz_010, xyz_011, xyz_100, xyz_101,
                       xyz_110, xyz_111},
                      1);
}

torch::Tensor cal_tri_inter_coef(const torch::Tensor &xyz_weight) {
  auto xyz_weight_x = xyz_weight.select(1, 0);
  auto xyz_weight_y = xyz_weight.select(1, 1);
  auto xyz_weight_z = xyz_weight.select(1, 2);
  auto coef_000 = (1 - xyz_weight_x) * (1 - xyz_weight_y) * (1 - xyz_weight_z);
  auto coef_001 = (1 - xyz_weight_x) * (1 - xyz_weight_y) * xyz_weight_z;
  auto coef_010 = (1 - xyz_weight_x) * xyz_weight_y * (1 - xyz_weight_z);
  auto coef_011 = (1 - xyz_weight_x) * xyz_weight_y * xyz_weight_z;
  auto coef_100 = xyz_weight_x * (1 - xyz_weight_y) * (1 - xyz_weight_z);
  auto coef_101 = xyz_weight_x * (1 - xyz_weight_y) * xyz_weight_z;
  auto coef_110 = xyz_weight_x * xyz_weight_y * (1 - xyz_weight_z);
  auto coef_111 = xyz_weight_x * xyz_weight_y * xyz_weight_z;
  return torch::stack({coef_000, coef_001, coef_010, coef_011, coef_100,
                       coef_101, coef_110, coef_111},
                      1);
}

torch::Tensor cal_inter_pair_coef(const torch::Tensor &vertex_sdf,
                                  const torch::Tensor &face_edge_pair_index,
                                  float iso_value) {
  auto face_edge_pair_sdf =
      vertex_sdf.index_select(0, face_edge_pair_index.view({-1})).view({-1, 2});
  auto face_edge_coef =
      (iso_value - face_edge_pair_sdf.select(1, 0)) /
      (face_edge_pair_sdf.select(1, 1) - face_edge_pair_sdf.select(1, 0));
  return face_edge_coef.view({-1, 3});
}

// Function to convert normalized quaternion to rotation matrix
torch::Tensor normalized_quat_to_rotmat(const torch::Tensor &quat) {
  // Ensure the input tensor has the correct shape
  TORCH_CHECK(quat.size(-1) == 4,
              "Quaternion must have 4 elements in the last dimension");

  // Unbind the quaternion into its components
  auto w = quat.select(-1, 0);
  auto x = quat.select(-1, 1);
  auto y = quat.select(-1, 2);
  auto z = quat.select(-1, 3);

  // Compute the rotation matrix components
  auto mat = torch::stack(
      {1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
       2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
       2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)},
      -1);

  // Reshape the result to the desired shape (..., 3, 3)
  return mat.reshape({-1, 3, 3});
}

torch::Tensor quat_to_rot(const torch::Tensor &quat, const bool &xyzw) {
  // https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
  auto rot = torch::zeros({3, 3}, quat.dtype()).to(quat.device());
  if (xyzw) {
    auto wxyz = torch::cat({quat.slice(0, 3, 4), quat.slice(0, 0, 3)}, 0);
    // auto q = quat.clone().to(torch::kFloat64);
    auto n = torch::dot(wxyz, wxyz);
    if (n.item<double>() < 1e-10) {
      return torch::eye(4, torch::kFloat64).to(quat.device());
    }
    wxyz *= torch::sqrt(2.0 / n);
    wxyz = torch::outer(wxyz, wxyz);
    rot[0][0] = 1.0 - wxyz[2][2] - wxyz[3][3];
    rot[0][1] = wxyz[1][2] - wxyz[3][0];
    rot[0][2] = wxyz[1][3] + wxyz[2][0];
    rot[1][0] = wxyz[1][2] + wxyz[3][0];
    rot[1][1] = 1.0 - wxyz[1][1] - wxyz[3][3];
    rot[1][2] = wxyz[2][3] - wxyz[1][0];
    rot[2][0] = wxyz[1][3] - wxyz[2][0];
    rot[2][1] = wxyz[2][3] + wxyz[1][0];
    rot[2][2] = 1.0 - wxyz[1][1] - wxyz[2][2];

    // rot[0][0] = 1 - 2.0 * (quat[1] * quat[1] + quat[2] * quat[2]);
    // rot[0][1] = 2.0 * (quat[0] * quat[1] - quat[2] * quat[3]);
    // rot[0][2] = 2.0 * (quat[0] * quat[2] + quat[1] * quat[3]);
    // rot[1][0] = 2.0 * (quat[0] * quat[1] + quat[2] * quat[3]);
    // rot[1][1] = 1 - 2.0 * (quat[0] * quat[0] + quat[2] * quat[2]);
    // rot[1][2] = 2.0 * (quat[1] * quat[2] - quat[0] * quat[3]);
    // rot[2][0] = 2.0 * (quat[0] * quat[2] - quat[1] * quat[3]);
    // rot[2][1] = 2.0 * (quat[1] * quat[2] + quat[0] * quat[3]);
    // rot[2][2] = 1 - 2.0 * (quat[0] * quat[0] + quat[1] * quat[1]);
  } else {
    rot[0][0] = 1 - 2.0 * (quat[2] * quat[2] + quat[3] * quat[3]);
    rot[0][1] = 2.0 * (quat[1] * quat[2] - quat[3] * quat[0]);
    rot[0][2] = 2.0 * (quat[1] * quat[3] + quat[2] * quat[0]);
    rot[1][0] = 2.0 * (quat[1] * quat[2] + quat[3] * quat[0]);
    rot[1][1] = 1 - 2.0 * (quat[1] * quat[1] + quat[3] * quat[3]);
    rot[1][2] = 2.0 * (quat[2] * quat[3] - quat[1] * quat[0]);
    rot[2][0] = 2.0 * (quat[1] * quat[3] - quat[2] * quat[0]);
    rot[2][1] = 2.0 * (quat[2] * quat[3] + quat[1] * quat[0]);
    rot[2][2] = 1 - 2.0 * (quat[1] * quat[1] + quat[2] * quat[2]);
  }
  return rot;
}

torch::Tensor rot_to_quat(const torch::Tensor &rotation) {
  auto Rxx = rotation[0][0];
  auto Ryx = rotation[0][1];
  auto Rzx = rotation[0][2];
  auto Rxy = rotation[1][0];
  auto Ryy = rotation[1][1];
  auto Rzy = rotation[1][2];
  auto Rxz = rotation[2][0];
  auto Ryz = rotation[2][1];
  auto Rzz = rotation[2][2];

  torch::Tensor K = torch::zeros({4, 4}, rotation.options());
  K[0][0] = Rxx - Ryy - Rzz;
  K[1][0] = Ryx + Rxy;
  K[1][1] = Ryy - Rxx - Rzz;
  K[2][0] = Rzx + Rxz;
  K[2][1] = Rzy + Ryz;
  K[2][2] = Rzz - Rxx - Ryy;
  K[3][0] = Ryz - Rzy;
  K[3][1] = Rzx - Rxz;
  K[3][2] = Rxy - Ryx;
  K[3][3] = Rxx + Ryy + Rzz;
  K /= 3.0;

  auto eig_results = torch::linalg::eigh(K, "L");
  auto eigvals = std::get<0>(eig_results);
  auto eigvecs = std::get<1>(eig_results);

  auto max_eigval = eigvals.argmax();

  auto qvec = torch::zeros(4, rotation.options());
  qvec[0] = eigvecs[3][max_eigval];
  qvec[1] = eigvecs[0][max_eigval];
  qvec[2] = eigvecs[1][max_eigval];
  qvec[3] = eigvecs[2][max_eigval];
  if (qvec[0].item<float>() < 0) {
    qvec *= -1;
  }

  return qvec;

  // // w,x,y,z
  // auto quat = torch::zeros(4, rotation.dtype()).to(rotation.device());
  // quat[0] =
  //     0.5 * torch::sqrt(1 + rotation[0][0] + rotation[1][1] +
  //     rotation[2][2]);
  // quat[1] = 0.25 * (rotation[2][1] - rotation[1][2]) / quat[0];
  // quat[2] = 0.25 * (rotation[0][2] - rotation[2][0]) / quat[0];
  // quat[3] = 0.25 * (rotation[1][0] - rotation[0][1]) / quat[0];
  // return quat;
}

torch::Tensor positional_encode(const torch::Tensor &xyz) {
  // torch::Tensor xyz_pos_enc = xyz;
  // torch::Tensor xyz_expo = xyz.clone();
  // for (int i = 0; i <= 4; i++) {
  //   xyz_pos_enc = torch::cat(
  //       {xyz_pos_enc, torch::sin(xyz_expo), torch::cos(xyz_expo)}, -1);
  //   xyz_expo = 2 * xyz_expo;
  // }
  // return xyz_pos_enc;
  return torch::cat({xyz, torch::sin(xyz), torch::cos(xyz), torch::sin(2 * xyz),
                     torch::cos(2 * xyz), torch::sin(4 * xyz),
                     torch::cos(4 * xyz), torch::sin(8 * xyz),
                     torch::cos(8 * xyz), torch::sin(16 * xyz),
                     torch::cos(16 * xyz)},
                    -1);
}

torch::Tensor meshgrid_3d(float x_min, float x_max, float y_min, float y_max,
                          float z_min, float z_max, float resolution,
                          torch::Device &device) {
  auto x = torch::arange(x_min, x_max, resolution, device);
  auto y = torch::arange(y_min, y_max, resolution, device);

  torch::Tensor z;
  if (z_max <= z_min) {
    z = torch::tensor(z_min, device);
  } else {
    z = torch::arange(z_min, z_max, resolution, device);
  }
  auto xyz_seperate = torch::meshgrid({x, y, z});
  return torch::cat({xyz_seperate[0].unsqueeze(-1),
                     xyz_seperate[1].unsqueeze(-1),
                     xyz_seperate[2].unsqueeze(-1)},
                    -1);
}

torch::Tensor rotation_6d_to_matrix(const torch::Tensor &rotation6d) {
  /* """
  Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
  using Gram--Schmidt orthogonalization per Section B of [1]. Adapted from
  pytorch3d. Args: d6: 6D rotation representation, of size (*, 6)

  Returns:
      batch of rotation matrices of size (*, 3, 3)

  [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
  On the Continuity of Rotation Representations in Neural Networks.
  IEEE Conference on Computer Vision and Pattern Recognition, 2019.
  Retrieved from http://arxiv.org/abs/1812.07035
  """ */
  auto a1 = rotation6d.slice(-1, 0, 3);
  auto a2 = rotation6d.slice(-1, 3, 6);
  auto b1 = torch::nn::functional::normalize(
      a1, torch::nn::functional::NormalizeFuncOptions().dim(-1));
  auto b2 = a2 - (b1 * a2).sum(-1, true) * b1;

  b2 = torch::nn::functional::normalize(
      b2, torch::nn::functional::NormalizeFuncOptions().dim(-1));
  auto b3 = b1.cross(b2, -1);

  // [N,3,3]
  return torch::stack({b1, b2, b3}, -1);
}

#ifdef ENABLE_ROS
sensor_msgs::PointCloud2 tensor_to_pointcloud_msg(const torch::Tensor &_xyz,
                                                  const torch::Tensor &_rgb) {
  sensor_msgs::PointCloud2 pcl_msg;
  pcl_msg.height = 1;
  pcl_msg.width = _xyz.size(0);
  pcl_msg.is_dense = true;
  pcl_msg.is_bigendian = false;
  pcl_msg.point_step = 32;
  pcl_msg.row_step = pcl_msg.point_step * pcl_msg.width;
  pcl_msg.fields.resize(4);
  pcl_msg.fields[0].name = "x";
  pcl_msg.fields[0].offset = 0;
  pcl_msg.fields[0].datatype = sensor_msgs::PointField::FLOAT32;
  pcl_msg.fields[0].count = 1;
  pcl_msg.fields[1].name = "y";
  pcl_msg.fields[1].offset = 4;
  pcl_msg.fields[1].datatype = sensor_msgs::PointField::FLOAT32;
  pcl_msg.fields[1].count = 1;
  pcl_msg.fields[2].name = "z";
  pcl_msg.fields[2].offset = 8;
  pcl_msg.fields[2].datatype = sensor_msgs::PointField::FLOAT32;
  pcl_msg.fields[2].count = 1;
  pcl_msg.fields[3].name = "rgb";
  pcl_msg.fields[3].offset = 16;
  pcl_msg.fields[3].datatype = sensor_msgs::PointField::FLOAT32;
  pcl_msg.fields[3].count = 1;
  pcl_msg.data.resize(pcl_msg.row_step * pcl_msg.height);

  auto xyz_float = _xyz.to(torch::kFloat);
  auto options = xyz_float.options();
  auto tensor_cat =
      torch::cat({xyz_float, torch::empty({xyz_float.size(0), 1}, options),
                  encode_rgb_to_single_float(_rgb),
                  torch::empty({xyz_float.size(0), 3}, options)},
                 1);

  memcpy(pcl_msg.data.data(), tensor_cat.cpu().data_ptr(),
         tensor_cat.numel() * sizeof(float));
  return pcl_msg;
}

sensor_msgs::Image tensor_to_img_msg(const torch::Tensor &_image) {
  auto image_cv = tensor_to_cv_mat(_image);
  if (image_cv.type() == CV_16UC1) {
    image_cv = apply_colormap_to_depth(image_cv);
  }
  sensor_msgs::Image img_msg;
  img_msg.height = image_cv.rows;
  img_msg.width = image_cv.cols;
  if (image_cv.type() == CV_8UC3)
    img_msg.encoding = "bgr8";
  else if (image_cv.type() == CV_16UC1)
    img_msg.encoding = "mono16";
  else
    throw std::runtime_error("Not implemented");
  img_msg.is_bigendian = false;
  img_msg.step = image_cv.cols * image_cv.elemSize();
  img_msg.data.resize(img_msg.step * img_msg.height);
  memcpy(img_msg.data.data(), image_cv.data, img_msg.data.size());
  return img_msg;
}

visualization_msgs::Marker get_vix_voxel_map(const torch::Tensor &_xyz,
                                             float voxel_size, float r, float g,
                                             float b) {
  auto xyz_idx_uniq = (_xyz.squeeze() / voxel_size).floor().to(torch::kLong);
  xyz_idx_uniq = std::get<0>(torch::unique_dim(xyz_idx_uniq, 0));
  /// [n, 8, 3]
  xyz_idx_uniq = spc_ops::points_to_corners(xyz_idx_uniq);
  auto voxel_vertex_xyz = xyz_idx_uniq * voxel_size;
  /// [n, 24, 3]
  voxel_vertex_xyz =
      torch::stack(
          {voxel_vertex_xyz.select(1, 0), voxel_vertex_xyz.select(1, 1),
           voxel_vertex_xyz.select(1, 1), voxel_vertex_xyz.select(1, 3),
           voxel_vertex_xyz.select(1, 3), voxel_vertex_xyz.select(1, 2),
           voxel_vertex_xyz.select(1, 2), voxel_vertex_xyz.select(1, 0),
           voxel_vertex_xyz.select(1, 4), voxel_vertex_xyz.select(1, 5),
           voxel_vertex_xyz.select(1, 5), voxel_vertex_xyz.select(1, 7),
           voxel_vertex_xyz.select(1, 7), voxel_vertex_xyz.select(1, 6),
           voxel_vertex_xyz.select(1, 6), voxel_vertex_xyz.select(1, 4),
           voxel_vertex_xyz.select(1, 0), voxel_vertex_xyz.select(1, 4),
           voxel_vertex_xyz.select(1, 1), voxel_vertex_xyz.select(1, 5),
           voxel_vertex_xyz.select(1, 2), voxel_vertex_xyz.select(1, 6),
           voxel_vertex_xyz.select(1, 3), voxel_vertex_xyz.select(1, 7)},
          1)
          .to(torch::kDouble)
          .cpu();

  visualization_msgs::Marker marker;
  marker.type = visualization_msgs::Marker::LINE_LIST;
  marker.action = visualization_msgs::Marker::ADD;
  marker.color.a = 1.0;
  marker.color.r = r;
  marker.color.g = g;
  marker.color.b = b;
  marker.scale.x = 0.1 * voxel_size;
  marker.points.resize(voxel_vertex_xyz.size(0) * 24);
  memcpy(marker.points.data(), voxel_vertex_xyz.data_ptr<double>(),
         voxel_vertex_xyz.numel() * sizeof(double));
  return marker;
}

visualization_msgs::Marker get_vis_shift_map(torch::Tensor _pos_W_M,
                                             float _x_min, float _x_max,
                                             float _y_min, float _y_max,
                                             float _z_min, float _z_max) {
  auto t_W_M_cpu = _pos_W_M.cpu();
  auto t_W_M_a = t_W_M_cpu.accessor<float, 2>();

  visualization_msgs::Marker marker;
  marker.points.resize(24);
  marker.type = visualization_msgs::Marker::LINE_LIST;
  marker.action = visualization_msgs::Marker::ADD;
  marker.color.a = 1.0;
  marker.color.r = 1.0;
  marker.color.g = 1.0;
  marker.color.b = 1.0;
  marker.scale.x = 0.01;
  {
    marker.points[0].x = t_W_M_a[0][0] + _x_min;
    marker.points[0].y = t_W_M_a[0][1] + _y_min;
    marker.points[0].z = t_W_M_a[0][2] + _z_min;
    marker.points[1].x = t_W_M_a[0][0] + _x_max;
    marker.points[1].y = t_W_M_a[0][1] + _y_min;
    marker.points[1].z = t_W_M_a[0][2] + _z_min;

    marker.points[2].x = t_W_M_a[0][0] + _x_max;
    marker.points[2].y = t_W_M_a[0][1] + _y_min;
    marker.points[2].z = t_W_M_a[0][2] + _z_min;
    marker.points[3].x = t_W_M_a[0][0] + _x_max;
    marker.points[3].y = t_W_M_a[0][1] + _y_max;
    marker.points[3].z = t_W_M_a[0][2] + _z_min;

    marker.points[4].x = t_W_M_a[0][0] + _x_max;
    marker.points[4].y = t_W_M_a[0][1] + _y_max;
    marker.points[4].z = t_W_M_a[0][2] + _z_min;
    marker.points[5].x = t_W_M_a[0][0] + _x_min;
    marker.points[5].y = t_W_M_a[0][1] + _y_max;
    marker.points[5].z = t_W_M_a[0][2] + _z_min;

    marker.points[6].x = t_W_M_a[0][0] + _x_min;
    marker.points[6].y = t_W_M_a[0][1] + _y_max;
    marker.points[6].z = t_W_M_a[0][2] + _z_min;
    marker.points[7].x = t_W_M_a[0][0] + _x_min;
    marker.points[7].y = t_W_M_a[0][1] + _y_min;
    marker.points[7].z = t_W_M_a[0][2] + _z_min;

    marker.points[8].x = t_W_M_a[0][0] + _x_min;
    marker.points[8].y = t_W_M_a[0][1] + _y_min;
    marker.points[8].z = t_W_M_a[0][2] + _z_max;
    marker.points[9].x = t_W_M_a[0][0] + _x_max;
    marker.points[9].y = t_W_M_a[0][1] + _y_min;
    marker.points[9].z = t_W_M_a[0][2] + _z_max;

    marker.points[10].x = t_W_M_a[0][0] + _x_max;
    marker.points[10].y = t_W_M_a[0][1] + _y_min;
    marker.points[10].z = t_W_M_a[0][2] + _z_max;
    marker.points[11].x = t_W_M_a[0][0] + _x_max;
    marker.points[11].y = t_W_M_a[0][1] + _y_max;
    marker.points[11].z = t_W_M_a[0][2] + _z_max;

    marker.points[12].x = t_W_M_a[0][0] + _x_max;
    marker.points[12].y = t_W_M_a[0][1] + _y_max;
    marker.points[12].z = t_W_M_a[0][2] + _z_max;
    marker.points[13].x = t_W_M_a[0][0] + _x_min;
    marker.points[13].y = t_W_M_a[0][1] + _y_max;
    marker.points[13].z = t_W_M_a[0][2] + _z_max;

    marker.points[14].x = t_W_M_a[0][0] + _x_min;
    marker.points[14].y = t_W_M_a[0][1] + _y_max;
    marker.points[14].z = t_W_M_a[0][2] + _z_max;
    marker.points[15].x = t_W_M_a[0][0] + _x_min;
    marker.points[15].y = t_W_M_a[0][1] + _y_min;
    marker.points[15].z = t_W_M_a[0][2] + _z_max;

    marker.points[16].x = t_W_M_a[0][0] + _x_min;
    marker.points[16].y = t_W_M_a[0][1] + _y_min;
    marker.points[16].z = t_W_M_a[0][2] + _z_min;
    marker.points[17].x = t_W_M_a[0][0] + _x_min;
    marker.points[17].y = t_W_M_a[0][1] + _y_min;
    marker.points[17].z = t_W_M_a[0][2] + _z_max;

    marker.points[18].x = t_W_M_a[0][0] + _x_max;
    marker.points[18].y = t_W_M_a[0][1] + _y_min;
    marker.points[18].z = t_W_M_a[0][2] + _z_min;
    marker.points[19].x = t_W_M_a[0][0] + _x_max;
    marker.points[19].y = t_W_M_a[0][1] + _y_min;
    marker.points[19].z = t_W_M_a[0][2] + _z_max;

    marker.points[20].x = t_W_M_a[0][0] + _x_max;
    marker.points[20].y = t_W_M_a[0][1] + _y_max;
    marker.points[20].z = t_W_M_a[0][2] + _z_min;
    marker.points[21].x = t_W_M_a[0][0] + _x_max;
    marker.points[21].y = t_W_M_a[0][1] + _y_max;
    marker.points[21].z = t_W_M_a[0][2] + _z_max;

    marker.points[22].x = t_W_M_a[0][0] + _x_min;
    marker.points[22].y = t_W_M_a[0][1] + _y_max;
    marker.points[22].z = t_W_M_a[0][2] + _z_min;
    marker.points[23].x = t_W_M_a[0][0] + _x_min;
    marker.points[23].y = t_W_M_a[0][1] + _y_max;
    marker.points[23].z = t_W_M_a[0][2] + _z_max;
  }
  return marker;
}
#endif

} // namespace utils