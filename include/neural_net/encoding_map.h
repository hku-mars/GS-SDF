#pragma once

#include "neural_net/sub_map.h"
#include "tcnn_binding/tcnn_binding.h"

struct EncodingMap : SubMap {
  EncodingMap(const torch::Tensor &_pos_W_M, std::string _name, float _x_min,
              float _x_max, float _y_min, float _y_max, float _z_min,
              float _z_max);
  std::string name_;
  std::shared_ptr<TCNNEncoding> p_encoder_tcnn_;

  std::mutex state_mutex_;

  torch::Tensor encoding(const torch::Tensor &xyz, const int &encoding_type = 0,
                         const bool &normalized = false);

  void activate();

  void freeze();
};