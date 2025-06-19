#pragma once

#include "kaolin_wisp_cpp/octree_as/octree_as.h"

struct SubMap : torch::nn::Module {
  typedef std::shared_ptr<SubMap> Ptr;
  SubMap(const torch::Tensor &_pos_W_M, float _x_min, float _x_max,
         float _y_min, float _y_max, float _z_min, float _z_max);

  std::shared_ptr<OctreeAS> p_acc_strcut_occ_;

  torch::Tensor pos_W_M_; // pos: Map(xyz) to World
  torch::Tensor xyz_max_W_,
      xyz_min_W_;                       // relative to World
  torch::Tensor xyz_max_M_, xyz_min_M_; // relative to Map
  torch::Tensor xyz_max_M_margin_, xyz_min_M_margin_;

  bool active_;

  void update_octree_as(const torch::Tensor &_xyz,
                        const bool &_is_prior = false, const int &_type = 0);

  torch::Tensor get_inrange_mask(const torch::Tensor &_xyz,
                                 float padding = 0.0) const;

  void get_intersect_point(const torch::Tensor &_points,
                           const torch::Tensor &_rays, torch::Tensor &z_nears,
                           torch::Tensor &z_fars, torch::Tensor &mask_intersect,
                           float padding = 0.0) const;

  torch::Tensor get_valid_mask(const torch::Tensor &xyz, int level = -1);

  torch::Tensor xyz_to_m1p1_pts(const torch::Tensor &_xyz);
  torch::Tensor m1p1_pts_to_xyz(const torch::Tensor &_m1p1_pts);
  static torch::Tensor scale_from_m1p1(const torch::Tensor &_m1p1_tensor);
  static torch::Tensor scale_to_m1p1(const torch::Tensor &_tensor);

  torch::Tensor xyz_to_zp1_pts(const torch::Tensor &_xyz);
  torch::Tensor zp1_pts_to_xyz(const torch::Tensor &_zp1_pts);
  static torch::Tensor scale_from_zp1(const torch::Tensor &_zp1_tensor);
  static torch::Tensor scale_to_zp1(const torch::Tensor &_ntensor);
  float scale_to_zp1(const float &_value);
};