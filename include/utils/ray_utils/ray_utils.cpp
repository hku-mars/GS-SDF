#include "ray_utils.h"

int64_t RaySamples::size(const int &dim) const { return origin.size(dim); }

torch::Device RaySamples::device() const { return origin.device(); }

RaySamples RaySamples::index(
    const torch::ArrayRef<at::indexing::TensorIndex> &index) const {
  return {origin.numel() > 0 ? origin.index(index) : torch::Tensor(),
          direction.numel() > 0 ? direction.index(index) : torch::Tensor(),
          zdir_norm.numel() > 0 ? zdir_norm.index(index) : torch::Tensor(),

          pred_xyz.numel() > 0 ? pred_xyz.index(index) : torch::Tensor(),
          pred_sdf.numel() > 0 ? pred_sdf.index(index) : torch::Tensor(),
          pred_isigma.numel() > 0 ? pred_isigma.index(index) : torch::Tensor(),
          ridx.numel() > 0 ? ridx.index(index) : torch::Tensor()};
}

RaySamples RaySamples::index_select(const int64_t dim,
                                    const torch::Tensor &index) const {
  return {origin.numel() > 0 ? origin.index_select(dim, index)
                             : torch::Tensor(),
          direction.numel() > 0 ? direction.index_select(dim, index)
                                : torch::Tensor(),
          zdir_norm.numel() > 0 ? zdir_norm.index_select(dim, index)
                                : torch::Tensor(),

          pred_xyz.numel() > 0 ? pred_xyz.index_select(dim, index)
                               : torch::Tensor(),
          pred_sdf.numel() > 0 ? pred_sdf.index_select(dim, index)
                               : torch::Tensor(),
          pred_isigma.numel() > 0 ? pred_isigma.index_select(dim, index)
                                  : torch::Tensor(),
          ridx.numel() > 0 ? ridx.index_select(dim, index) : torch::Tensor()};
}

RaySamples RaySamples::slice(int64_t dim, c10::optional<int64_t> start,
                             c10::optional<int64_t> end, int64_t step) const {
  return {origin.numel() > 0 ? origin.slice(dim, start, end, step)
                             : torch::Tensor(),
          direction.numel() > 0 ? direction.slice(dim, start, end, step)
                                : torch::Tensor(),
          zdir_norm.numel() > 0 ? zdir_norm.slice(dim, start, end, step)
                                : torch::Tensor(),

          pred_xyz.numel() > 0 ? pred_xyz.slice(dim, start, end, step)
                               : torch::Tensor(),
          pred_sdf.numel() > 0 ? pred_sdf.slice(dim, start, end, step)
                               : torch::Tensor(),
          pred_isigma.numel() > 0 ? pred_isigma.slice(dim, start, end, step)
                                  : torch::Tensor(),
          ridx.numel() > 0 ? ridx.slice(dim, start, end, step)
                           : torch::Tensor()};
}

RaySamples RaySamples::cat(const RaySamples &other) const {
  return {
      origin.numel() > 0 ? torch::cat({origin, other.origin}, 0)
                         : torch::Tensor(),
      direction.numel() > 0 ? torch::cat({direction, other.direction}, 0)
                            : torch::Tensor(),
      zdir_norm.numel() > 0 ? torch::cat({zdir_norm, other.zdir_norm}, 0)
                            : torch::Tensor(),

      pred_xyz.numel() > 0 ? torch::cat({pred_xyz, other.pred_xyz}, 0)
                           : torch::Tensor(),
      pred_sdf.numel() > 0 ? torch::cat({pred_sdf, other.pred_sdf}, 0)
                           : torch::Tensor(),
      pred_isigma.numel() > 0 ? torch::cat({pred_isigma, other.pred_isigma}, 0)
                              : torch::Tensor(),
      ridx.numel() > 0 ? torch::cat({ridx, other.ridx}, 0) : torch::Tensor()};
}

RaySamples RaySamples::to(const torch::TensorOptions &option) const {
  return {origin.numel() > 0 ? origin.to(option) : torch::Tensor(),
          direction.numel() > 0 ? direction.to(option) : torch::Tensor(),
          zdir_norm.numel() > 0 ? zdir_norm.to(option) : torch::Tensor(),

          pred_xyz.numel() > 0 ? pred_xyz.to(option) : torch::Tensor(),
          pred_sdf.numel() > 0 ? pred_sdf.to(option) : torch::Tensor(),
          pred_isigma.numel() > 0 ? pred_isigma.to(option) : torch::Tensor(),
          ridx.numel() > 0 ? ridx.to(option) : torch::Tensor()};
}

RaySamples RaySamples::contiguous() const {
  return {origin.numel() > 0 ? origin.contiguous() : torch::Tensor(),
          direction.numel() > 0 ? direction.contiguous() : torch::Tensor(),
          zdir_norm.numel() > 0 ? zdir_norm.contiguous() : torch::Tensor(),

          pred_xyz.numel() > 0 ? pred_xyz.contiguous() : torch::Tensor(),
          pred_sdf.numel() > 0 ? pred_sdf.contiguous() : torch::Tensor(),
          pred_isigma.numel() > 0 ? pred_isigma.contiguous() : torch::Tensor(),
          ridx.numel() > 0 ? ridx.contiguous() : torch::Tensor()};
}

RaySamples RaySamples::clone() const {
  return {
      origin.numel() > 0 ? origin.clone() : torch::Tensor(),
      direction.numel() > 0 ? direction.clone() : torch::Tensor(),
      zdir_norm.numel() > 0 ? zdir_norm.clone() : torch::Tensor(),

      pred_xyz.numel() > 0 ? pred_xyz.clone() : torch::Tensor(),
      pred_sdf.numel() > 0 ? pred_sdf.clone() : torch::Tensor(),
      pred_isigma.numel() > 0 ? pred_isigma.clone() : torch::Tensor(),
      ridx.numel() > 0 ? ridx.clone() : torch::Tensor(),
  };
}

DepthSamples DepthSamples::index(
    const torch::ArrayRef<at::indexing::TensorIndex> &index) const {
  return {
      RaySamples::index(index),
      xyz.numel() > 0 ? xyz.index(index) : torch::Tensor(),
      depth.numel() > 0 ? depth.index(index) : torch::Tensor(),

      ray_sdf.numel() > 0 ? ray_sdf.index(index) : torch::Tensor(),
  };
}

DepthSamples DepthSamples::index_select(const int64_t dim,
                                        const torch::Tensor &index) const {
  return {RaySamples::index_select(dim, index),
          xyz.numel() > 0 ? xyz.index_select(dim, index) : torch::Tensor(),
          depth.numel() > 0 ? depth.index_select(dim, index) : torch::Tensor(),

          ray_sdf.numel() > 0 ? ray_sdf.index_select(dim, index)
                              : torch::Tensor()};
}

DepthSamples DepthSamples::slice(int64_t dim, c10::optional<int64_t> start,
                                 c10::optional<int64_t> end,
                                 int64_t step) const {
  return {RaySamples::slice(dim, start, end, step),
          xyz.numel() > 0 ? xyz.slice(dim, start, end, step) : torch::Tensor(),
          depth.numel() > 0 ? depth.slice(dim, start, end, step)
                            : torch::Tensor(),

          ray_sdf.numel() > 0 ? ray_sdf.slice(dim, start, end, step)
                              : torch::Tensor()};
}

DepthSamples DepthSamples::cat(const DepthSamples &other) const {
  return {RaySamples::cat(other),
          xyz.numel() > 0 ? torch::cat({xyz, other.xyz}, 0) : torch::Tensor(),
          depth.numel() > 0 ? torch::cat({depth, other.depth}, 0)
                            : torch::Tensor(),

          ray_sdf.numel() > 0 ? torch::cat({ray_sdf, other.ray_sdf}, 0)
                              : torch::Tensor()};
}

DepthSamples DepthSamples::to(const torch::TensorOptions &option) const {
  return {RaySamples::to(option),
          xyz.numel() > 0 ? xyz.to(option) : torch::Tensor(),
          depth.numel() > 0 ? depth.to(option) : torch::Tensor(),

          ray_sdf.numel() > 0 ? ray_sdf.to(option) : torch::Tensor()};
}

DepthSamples DepthSamples::contiguous() const {
  return {RaySamples::contiguous(),
          xyz.numel() > 0 ? xyz.contiguous() : torch::Tensor(),
          depth.numel() > 0 ? depth.contiguous() : torch::Tensor(),

          ray_sdf.numel() > 0 ? ray_sdf.contiguous() : torch::Tensor()};
}

DepthSamples DepthSamples::clone() const {
  return {RaySamples::clone(), xyz.numel() > 0 ? xyz.clone() : torch::Tensor(),
          depth.numel() > 0 ? depth.clone() : torch::Tensor(),

          ray_sdf.numel() > 0 ? ray_sdf.clone() : torch::Tensor()};
}

ColorSamples ColorSamples::index(
    const torch::ArrayRef<at::indexing::TensorIndex> &index) const {
  return {
      RaySamples::index(index),
      rgb.numel() > 0 ? rgb.index(index) : torch::Tensor(),
  };
}

ColorSamples ColorSamples::index_select(const int64_t dim,
                                        const torch::Tensor &index) const {
  return {
      RaySamples::index_select(dim, index),
      rgb.numel() > 0 ? rgb.index_select(dim, index) : torch::Tensor(),
  };
}

ColorSamples ColorSamples::slice(int64_t dim, c10::optional<int64_t> start,
                                 c10::optional<int64_t> end,
                                 int64_t step) const {
  return {RaySamples::slice(dim, start, end, step),
          rgb.numel() > 0 ? rgb.slice(dim, start, end, step) : torch::Tensor()};
}

ColorSamples ColorSamples::cat(const ColorSamples &other) const {
  return {RaySamples::cat(other),
          rgb.numel() > 0 ? torch::cat({rgb, other.rgb}, 0) : torch::Tensor()};
}

ColorSamples ColorSamples::to(const torch::TensorOptions &option) const {
  return {RaySamples::to(option),
          rgb.numel() > 0 ? rgb.to(option) : torch::Tensor()};
}

ColorSamples ColorSamples::contiguous() const {
  return {RaySamples::contiguous(),
          rgb.numel() > 0 ? rgb.contiguous() : torch::Tensor()};
}

ColorSamples ColorSamples::clone() const {
  return {RaySamples::clone(), rgb.numel() > 0 ? rgb.clone() : torch::Tensor()};
}

int64_t RaySamplesVec::size(const int &dim) const { return origin.size(); }

void RaySamplesVec::emplace_back(const RaySamples &ray) {
  if (ray.origin.numel() > 0) {
    origin.emplace_back(ray.origin);
  }
  if (ray.direction.numel() > 0) {
    direction.emplace_back(ray.direction);
  }
  if (ray.zdir_norm.numel() > 0) {
    zdir_norm.emplace_back(ray.zdir_norm);
  }
  if (ray.pred_xyz.numel() > 0) {
    pred_xyz.emplace_back(ray.pred_xyz);
  }
  if (ray.pred_sdf.numel() > 0) {
    pred_sdf.emplace_back(ray.pred_sdf);
  }
  if (ray.pred_isigma.numel() > 0) {
    pred_isigma.emplace_back(ray.pred_isigma);
  }
}

void RaySamplesVec::clear() {
  origin.clear();
  direction.clear();
  zdir_norm.clear();
  pred_xyz.clear();
  pred_sdf.clear();
  pred_isigma.clear();
}

void RaySamplesVec::shrink_to_fit() {
  origin.shrink_to_fit();
  direction.shrink_to_fit();
  zdir_norm.shrink_to_fit();
  pred_xyz.shrink_to_fit();
  pred_sdf.shrink_to_fit();
  pred_isigma.shrink_to_fit();
}

void RaySamplesVec::release() {
  clear();
  shrink_to_fit();
}

RaySamples RaySamplesVec::cat() const {
  RaySamples out;
  if (!origin.empty()) {
    out.origin = torch::cat(origin, 0);
  }
  if (!direction.empty()) {
    out.direction = torch::cat(direction, 0);
  }
  if (!zdir_norm.empty()) {
    out.zdir_norm = torch::cat(zdir_norm, 0);
  }
  if (!pred_xyz.empty()) {
    out.pred_xyz = torch::cat(pred_xyz, 0);
  }
  if (!pred_sdf.empty()) {
    out.pred_sdf = torch::cat(pred_sdf, 0);
  }
  if (!pred_isigma.empty()) {
    out.pred_isigma = torch::cat(pred_isigma, 0);
  }
  return out;
}

void DepthSamplesVec::emplace_back(const DepthSamples &input) {
  RaySamplesVec::emplace_back(input);
  if (input.xyz.numel() > 0) {
    xyz.emplace_back(input.xyz);
  }
  if (input.depth.numel() > 0) {
    depth.emplace_back(input.depth);
  }
  if (input.ray_sdf.numel() > 0) {
    ray_sdf.emplace_back(input.ray_sdf);
  }
}

void DepthSamplesVec::clear() {
  RaySamplesVec::clear();
  xyz.clear();
  depth.clear();
  ray_sdf.clear();
}

void DepthSamplesVec::shrink_to_fit() {
  RaySamplesVec::shrink_to_fit();
  xyz.shrink_to_fit();
  depth.shrink_to_fit();
  ray_sdf.shrink_to_fit();
}

void DepthSamplesVec::release() {
  clear();
  shrink_to_fit();
}

DepthSamples DepthSamplesVec::cat() const {
  DepthSamples out = {RaySamplesVec::cat()};
  if (!xyz.empty()) {
    out.xyz = torch::cat(xyz, 0);
  }
  if (!depth.empty()) {
    out.depth = torch::cat(depth, 0);
  }
  if (!ray_sdf.empty()) {
    out.ray_sdf = torch::cat(ray_sdf, 0);
  }
  return out;
}

void ColorSamplesVec::emplace_back(const ColorSamples &input) {
  RaySamplesVec::emplace_back(input);
  if (input.rgb.numel() > 0) {
    rgb.emplace_back(input.rgb);
  }
}

void ColorSamplesVec::clear() {
  RaySamplesVec::clear();
  rgb.clear();
}

void ColorSamplesVec::shrink_to_fit() {
  RaySamplesVec::shrink_to_fit();
  rgb.shrink_to_fit();
}

void ColorSamplesVec::release() {
  clear();
  shrink_to_fit();
}

ColorSamples ColorSamplesVec::cat() const {
  ColorSamples out = {RaySamplesVec::cat()};
  if (!rgb.empty()) {
    out.rgb = torch::cat(rgb, 0);
  }
  return out;
}