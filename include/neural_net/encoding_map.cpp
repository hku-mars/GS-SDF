#include "encoding_map.h"
#include "params/params.h"

using namespace std;

EncodingMap::EncodingMap(const torch::Tensor &_pos_W_M, std::string _name,
                         float _x_min, float _x_max, float _y_min, float _y_max,
                         float _z_min, float _z_max)
    : SubMap(_pos_W_M, _x_min, _x_max, _y_min, _y_max, _z_min, _z_max),
      name_(_name) {
  // int base_res = 16;
  // auto growth_factor = exp((log(8192) - log(base_res)) / (k_n_levels - 1));

  // The number of encoded dimensions is n_levels * n_features_per_level
  nlohmann::json encoding_config = {
      {"otype", "Grid"},
      {"type", "Hash"},
      {"n_levels", k_n_levels},
      {"n_features_per_level", k_n_features_per_level},
      {"log2_hashmap_size", k_log2_hashmap_size},
      {"base_resolution", 32},
      {"per_level_scale", 2.0},
      {"interpolation", "Linear"}};

  p_encoder_tcnn_ =
      std::make_shared<TCNNEncoding>(3, encoding_config, "encoder_" + name_);

  active_ = true;
}

torch::Tensor EncodingMap::encoding(const torch::Tensor &xyz,
                                    const int &encoding_type,
                                    const bool &normalized) {
  // Make sure the tcnn gets inputs between 0 and 1.
  torch::Tensor normalized_xyz;
  if (normalized) {
    normalized_xyz = xyz;
  } else {
    normalized_xyz = xyz_to_zp1_pts(xyz);
  }

  /* TODO: contraction leads to NAN!!!!
  static bool scene_contraction = true;
  if (scene_contraction) {
    // static float zp1_leaf_size = k_leaf_size * k_map_size_inv;
    static float zp1_leaf_size = k_boundary_size * k_map_size_inv;
    static float inner_bound = 0.5f - zp1_leaf_size;
    auto tmp_xyz = normalized_xyz - 0.5f;
    auto tmp_xyz_norm =
        tmp_xyz.abs().detach() + 1e-6f; // avoid division by zero
    tmp_xyz = torch::where(
        tmp_xyz_norm < inner_bound, tmp_xyz,
        (inner_bound + zp1_leaf_size * (1 - inner_bound / tmp_xyz_norm)) *
            tmp_xyz / tmp_xyz_norm);
    normalized_xyz = tmp_xyz + 0.5f;
  } */

  // [n, feat_dim]
  return p_encoder_tcnn_->forward(normalized_xyz);
}

void EncodingMap::activate() {
  state_mutex_.lock();
  this->to(k_device);
  active_ = true;
  state_mutex_.unlock();
}

void EncodingMap::freeze() {
  state_mutex_.lock();
  this->to(torch::kCPU);
  active_ = false;
  state_mutex_.unlock();
}
