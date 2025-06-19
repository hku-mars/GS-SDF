#pragma once
#include "cameras.hpp"

namespace sensor {

struct Sensors {
  sensor::Cameras camera;

  // extrinsics
  torch::Tensor
      T_C_L; // [4, 4]; extrinsic param, transformation from lidar to camera
  torch::Tensor
      T_B_L; // [4, 4]; extrinsic param, transformation from lidar to body
  torch::Tensor
      T_B_C; // [4, 4]; extrinsic param, transformation from camera to body
};
} // namespace sensor
