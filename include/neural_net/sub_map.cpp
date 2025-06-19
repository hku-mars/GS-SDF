#include "sub_map.h"
#include "kaolin_wisp_cpp/spc_ops/spc_ops.h"
#include "params/params.h"

using namespace std;

SubMap::SubMap(const torch::Tensor &_pos_W_M, float _x_min, float _x_max,
               float _y_min, float _y_max, float _z_min, float _z_max)
    : torch::nn::Module(), pos_W_M_(_pos_W_M.view({1, 3}).to(k_device)) {
  xyz_min_M_ = torch::tensor({{_x_min, _y_min, _z_min}}, pos_W_M_.options());
  xyz_max_M_ = torch::tensor({{_x_max, _y_max, _z_max}}, pos_W_M_.options());

  xyz_min_W_ = pos_W_M_ + xyz_min_M_;
  xyz_max_W_ = pos_W_M_ + xyz_max_M_;

  /// visualization init
  xyz_min_M_margin_ = xyz_min_M_ + 0.5 * k_leaf_size;
  xyz_max_M_margin_ = xyz_max_M_ - 0.5 * k_leaf_size;
  active_ = true;
}

void SubMap::update_octree_as(const torch::Tensor &_xyz, const bool &_is_prior,
                              const int &_type) {
  // Make sure the kaolin gets inputs between -1 and 1.
  auto normalized_xyz = xyz_to_m1p1_pts(_xyz);
  auto qpts = spc_ops::quantize_points(normalized_xyz, k_octree_level);

  qpts = get<0>(torch::unique_dim(qpts.contiguous(), 0));
  if (!_is_prior) {
    int res = std::pow(2, k_octree_level);
    qpts = spc_ops::points_to_neighbors(qpts).view({-1, 3}).clamp(0, res - 1);
  }
  p_acc_strcut_occ_ =
      std::shared_ptr<OctreeAS>(from_quantized_points(qpts, k_octree_level));
}

torch::Tensor SubMap::get_inrange_mask(const torch::Tensor &_xyz,
                                       float padding) const {
  torch::Tensor max_mask =
      _xyz < (xyz_max_W_ - padding - 1e-6).to(_xyz.device());
  torch::Tensor min_mask =
      _xyz > (xyz_min_W_ + padding + 1e-6).to(_xyz.device());
  torch::Tensor mask = max_mask & min_mask;
  return mask.all(1);
}

void SubMap::get_intersect_point(const torch::Tensor &_points,
                                 const torch::Tensor &_rays,
                                 torch::Tensor &z_nears, torch::Tensor &z_fars,
                                 torch::Tensor &mask_intersect,
                                 float padding) const {
  // Force the parallel rays to intersect with plane, and they will be removed
  // by sanity check, add small number to avoid judgement
  torch::Tensor mask_parallels = _rays == 0;
  torch::Tensor tmp_rays =
      _rays.index_put({mask_parallels}, _rays.index({mask_parallels}) + 1e-6);

  // [n,3]
  torch::Tensor tmp_z_nears = (xyz_min_W_ + padding - _points) / tmp_rays;
  torch::Tensor tmp_z_fars = (xyz_max_W_ - padding - _points) / tmp_rays;

  // Make sure near is closer than far
  torch::Tensor mask_exchange = tmp_z_nears > tmp_z_fars;
  z_nears =
      tmp_z_nears.index_put({mask_exchange}, tmp_z_fars.index({mask_exchange}));
  z_fars =
      tmp_z_fars.index_put({mask_exchange}, tmp_z_nears.index({mask_exchange}));

  z_nears = get<0>(torch::max(z_nears, 1));
  z_fars = get<0>(torch::min(z_fars, 1));

  // check if intersect
  mask_intersect = z_nears < z_fars;
}

torch::Tensor SubMap::get_valid_mask(const torch::Tensor &xyz, int level) {
  // Make sure the kaolin gets inputs between -1 and 1.
  auto normalized_xyz = xyz_to_m1p1_pts(xyz);
  return p_acc_strcut_occ_->query(normalized_xyz, level).pidx > -1;
}

torch::Tensor SubMap::xyz_to_m1p1_pts(const torch::Tensor &_xyz) {
  return scale_to_m1p1(_xyz - pos_W_M_);
}
torch::Tensor SubMap::m1p1_pts_to_xyz(const torch::Tensor &_m1p1_pts) {
  return scale_from_m1p1(_m1p1_pts) + pos_W_M_;
}
torch::Tensor SubMap::scale_from_m1p1(const torch::Tensor &_m1p1_tensor) {
  return _m1p1_tensor * 0.5 * k_map_size;
}
torch::Tensor SubMap::scale_to_m1p1(const torch::Tensor &_ntensor) {
  return _ntensor * 2 * k_map_size_inv;
}

torch::Tensor SubMap::xyz_to_zp1_pts(const torch::Tensor &_xyz) {
  return 0.5f * xyz_to_m1p1_pts(_xyz) + 0.5f;
}
torch::Tensor SubMap::zp1_pts_to_xyz(const torch::Tensor &_zp1_pts) {
  return m1p1_pts_to_xyz(2.0f * _zp1_pts - 1.0f);
}
torch::Tensor SubMap::scale_from_zp1(const torch::Tensor &_zp1_tensor) {
  return _zp1_tensor * k_map_size;
}
torch::Tensor SubMap::scale_to_zp1(const torch::Tensor &_ntensor) {
  return _ntensor * k_map_size_inv;
}
float SubMap::scale_to_zp1(const float &_value) {
  return _value * k_map_size_inv;
}