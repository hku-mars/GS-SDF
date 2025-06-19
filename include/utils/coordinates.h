#pragma once

#include <torch/torch.h>

namespace coords {
/* clang-format off
https://docs.nerf.studio/quickstart/data_conventions.html#coordinate-conventions

# Camera/view coordinate systems
We use the COLMAP/OpenCV coordinate convention for cameras, where the Y and Z
axes are flipped from ours but the +X axis remains the same. Other codebases may
use the OpenGL/Blender (and original NeRF) convention, where the +X is right, +Y
is up, and +Z is pointing back and away from the camera. -Z is the look-at
direction.

# World coordinate systems
Our world space is oriented such that the up vector is +Z. The XY plane is
parallel to the ground plane. In the viewer, you’ll notice that red, green, and
blue vectors correspond to X, Y, and Z respectively.


-          | Camera                         | World
- OpenCV:    right handed, y-down, z-lookat | right handed, z-up
- COLMAP, Replica
- Blender:   right handed, y-up,  -z-lookat | right handed, z-up
- NerfStudio
- OpenGL:    right handed, y-up,  -z-lookat | right handed, y-up
- Kitti:     right handed, y-down, z-lookat | right handed, -y-up

Default coordinate systems: OpenCV
clang-format on */

enum SystemType { OpenCV = 0, Blender = 1, OpenGL = 2, Kitti = 3 };

torch::Tensor opencv_camera_coords();

torch::Tensor opencv_camera_transform();

torch::Tensor blender_camera_coords();

torch::Tensor blender_camera_transform();

torch::Tensor opencv_to_opengl_world_transform();

torch::Tensor opengl_to_opencv_world_rotation();
torch::Tensor opencv_to_opengl_camera_rotation();

torch::Tensor opengl_to_opencv_world_transform();

torch::Tensor rotpos_to_transform34(
    const torch::Tensor &rotation,
    const torch::Tensor &translation = torch::zeros({3, 1}, torch::kFloat32));

torch::Tensor rotpos_to_transform44(
    const torch::Tensor &rotation,
    const torch::Tensor &translation = torch::zeros({3, 1}, torch::kFloat32));

torch::Tensor change_camera_system(const torch::Tensor &pose,
                                   const int &pose_system_type);

torch::Tensor change_world_system(const torch::Tensor &pose,
                                  const int &_system_type);

torch::Tensor reset_world_system(const torch::Tensor &pose,
                                 const int &_system_type);

} // namespace coords