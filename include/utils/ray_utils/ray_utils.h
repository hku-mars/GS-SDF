#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <torch/torch.h>

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

struct RaySamples {
  torch::Tensor origin;    // [N, 3]
  torch::Tensor direction; // [N, 3]
  torch::Tensor zdir_norm; // [N, 1]

  torch::Tensor pred_xyz;    // [N,3]
  torch::Tensor pred_sdf;    // [N,1]
  torch::Tensor pred_isigma; // [N,1]

  torch::Tensor ridx; // [N]

  int64_t size(const int &dim = 0) const;

  torch::Device device() const;

  RaySamples
  index(const torch::ArrayRef<at::indexing::TensorIndex> &index) const;

  RaySamples index_select(const int64_t dim, const torch::Tensor &index) const;

  RaySamples slice(int64_t dim = 0, c10::optional<int64_t> start = c10::nullopt,
                   c10::optional<int64_t> end = c10::nullopt,
                   int64_t step = 1) const;

  RaySamples cat(const RaySamples &other) const;

  RaySamples to(const torch::TensorOptions &option) const;
  RaySamples contiguous() const;
  RaySamples clone() const;
};

struct DepthSamples : RaySamples {
  torch::Tensor xyz;   // [N, 3]
  torch::Tensor depth; // [N,1]

  torch::Tensor ray_sdf; // [N,1]

  DepthSamples
  index(const torch::ArrayRef<at::indexing::TensorIndex> &index) const;

  DepthSamples index_select(const int64_t dim,
                            const torch::Tensor &index) const;

  DepthSamples slice(int64_t dim = 0,
                     c10::optional<int64_t> start = c10::nullopt,
                     c10::optional<int64_t> end = c10::nullopt,
                     int64_t step = 1) const;

  DepthSamples cat(const DepthSamples &other) const;

  DepthSamples to(const torch::TensorOptions &option) const;

  DepthSamples contiguous() const;

  DepthSamples clone() const;
};

struct ColorSamples : RaySamples {

  torch::Tensor rgb; // [N,3]

  ColorSamples
  index(const torch::ArrayRef<at::indexing::TensorIndex> &index) const;

  ColorSamples index_select(const int64_t dim,
                            const torch::Tensor &index) const;

  ColorSamples slice(int64_t dim = 0,
                     c10::optional<int64_t> start = c10::nullopt,
                     c10::optional<int64_t> end = c10::nullopt,
                     int64_t step = 1) const;

  ColorSamples cat(const ColorSamples &other) const;

  ColorSamples to(const torch::TensorOptions &option) const;

  ColorSamples contiguous() const;

  ColorSamples clone() const;
};

struct RaySamplesVec {

  std::vector<torch::Tensor> origin;    // [N, 3]
  std::vector<torch::Tensor> direction; // [N, 3]
  std::vector<torch::Tensor> zdir_norm; // [N, 1]

  std::vector<torch::Tensor> pred_xyz;    // [N,3]
  std::vector<torch::Tensor> pred_sdf;    // [N,1]
  std::vector<torch::Tensor> pred_isigma; // [N,1]

  int64_t size(const int &dim = 0) const;

  void emplace_back(const RaySamples &ray);

  void clear();
  void shrink_to_fit();

  void release();

  RaySamples cat() const;
};

struct DepthSamplesVec : RaySamplesVec {
  std::vector<torch::Tensor> xyz;   // [N, 3]
  std::vector<torch::Tensor> depth; // [N,1]

  std::vector<torch::Tensor> ray_sdf; // [N,1]

  void emplace_back(const DepthSamples &input);

  void clear();

  void shrink_to_fit();

  void release();

  DepthSamples cat() const;
};

struct ColorSamplesVec : RaySamplesVec {
  std::vector<torch::Tensor> rgb; // [N,3]

  void emplace_back(const ColorSamples &input);

  void clear();

  void shrink_to_fit();

  void release();

  ColorSamples cat() const;
};