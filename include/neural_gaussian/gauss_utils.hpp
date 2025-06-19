#pragma once

#include <torch/torch.h>

// Function to get the projection matrix
torch::Tensor get_projection_matrix(float znear, float zfar, float fovX,
                                    float fovY, torch::Device device) {
  // """Create OpenGL-style projection matrix"""
  float tanHalfFovX = std::tan(fovX / 2.0f);
  float tanHalfFovY = std::tan(fovY / 2.0f);

  auto top = tanHalfFovY * znear;
  auto bottom = -top;
  auto right = tanHalfFovX * znear;
  auto left = -right;

  torch::Tensor P =
      torch::zeros({4, 4}, torch::dtype(torch::kFloat32).device(device));

  float z_sign = 1.0f;
  P[0][0] = 2.0 * znear / (right - left);
  P[1][1] = 2.0 * znear / (top - bottom);
  P[0][2] = (right + left) / (right - left);
  P[1][2] = (top + bottom) / (top - bottom);
  P[3][2] = z_sign;
  P[2][2] = z_sign * zfar / (zfar - znear);
  P[2][3] = -(zfar * znear) / (zfar - znear);
  return P;
}

// Function to generate a random quaternion tensor of shape (N, 4)
torch::Tensor random_quat_tensor(int64_t N) {
  // Generate random tensors u, v, w
  torch::Tensor u = torch::rand({N});
  torch::Tensor v = torch::rand({N});
  torch::Tensor w = torch::rand({N});

  // Compute the quaternion components
  torch::Tensor q1 = torch::sqrt(1 - u) * torch::sin(2 * M_PI * v);
  torch::Tensor q2 = torch::sqrt(1 - u) * torch::cos(2 * M_PI * v);
  torch::Tensor q3 = torch::sqrt(u) * torch::sin(2 * M_PI * w);
  torch::Tensor q4 = torch::sqrt(u) * torch::cos(2 * M_PI * w);

  // Stack the components to form the quaternion tensor
  return torch::stack({q1, q2, q3, q4}, -1);
}

/**
 * Returns the number of spherical harmonic bases for a given degree.
 * For spherical harmonics, the number of bases is (degree + 1)².
 *
 * @param degree The spherical harmonic degree, must be between 0-4 inclusive
 * @return The number of spherical harmonic bases
 * @throws std::invalid_argument if degree > 4
 */
inline constexpr int num_sh_bases(int degree) {
  // Check bounds at compile-time when possible
  if (degree < 0) {
    throw std::invalid_argument("Spherical harmonic degree cannot be negative");
  }

  if (degree > 4) {
    throw std::invalid_argument("Spherical harmonic degree cannot exceed 4");
  }

  // Using a lookup table for common values improves performance
  constexpr int sh_bases_lookup[] = {1, 4, 9, 16, 25};
  return sh_bases_lookup[degree];
}