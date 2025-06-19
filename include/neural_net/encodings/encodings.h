#pragma once

#include <torch/torch.h>

#include "tcnn_binding/tcnn_binding.h"
struct SHEncoding : TCNNEncoding {
  /* Spherical harmonic encoding */
  explicit SHEncoding(const int &_levels = 4) : levels_(_levels) {
    if (levels_ <= 0 || levels_ > 4) {
      throw std::invalid_argument("Spherical harmonic encoding only supports 1 "
                                  "to 4 levels, requested " +
                                  std::to_string(levels_));
    }
    nlohmann::json encoding_config = {
        {"otype", "SphericalHarmonics"},
        {"degree", levels_} // The SH degree up to which
                            // to evaluate the encoding.
                            // Produces degree^2 encoded
                            // dimensions.
    };

    init_encoding(3, encoding_config, "sh_encoding");
  }
  int levels_;

  size_t get_out_dim() const override { return levels_ * levels_; }
};