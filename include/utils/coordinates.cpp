#include "coordinates.h"

namespace coords {

torch::Tensor opencv_camera_coords() {
  /* return opencv to world coordinates rotation matrix */
  // clang-format off
  return torch::tensor({{ 0, 0, 1}, 
                        {-1, 0, 0}, 
                        { 0,-1, 0}}, torch::kFloat32);
  // clang-format on
}

torch::Tensor opencv_camera_transform() {
  // clang-format off
  return torch::tensor({{1, 0, 0, 0}, 
                        {0, 1, 0, 0}, 
                        {0, 0, 1, 0},
                        {0, 0, 0, 1}}, torch::kFloat32);
  // clang-format on
}

torch::Tensor blender_camera_coords() {
  /* return opencv to blender/opengl coordinates rotation matrix */
  // clang-format off
  return torch::tensor({{1, 0, 0}, 
                        {0, 0, 1}, 
                        {0,-1, 0}}, torch::kFloat32);
  // clang-format on
}

torch::Tensor blender_camera_transform() {
  /* return opencv to blender/opengl  coordinates rotation matrix */
  // clang-format off
  return torch::tensor({{1, 0, 0, 0}, 
                        {0, 0, 1, 0}, 
                        {0,-1, 0, 0},
                        {0, 0, 0, 1}}, torch::kFloat32);
  // clang-format on
}

torch::Tensor opencv_to_blender_camera_transform() {
  /* return opencv to blender/opengl coordinates transformation matrix */
  // clang-format off
  return torch::tensor({{1, 0, 0, 0}, 
                        {0,-1, 0, 0}, 
                        {0, 0,-1, 0},
                        {0, 0, 0, 1}}, torch::kFloat32);
  // clang-format on
}

torch::Tensor opencv_to_opengl_world_transform() {
  /* return opencv to blender/opengl coordinates transformation matrix */
  // clang-format off
  return opencv_to_blender_camera_transform();
  // clang-format on
}

torch::Tensor opengl_to_opencv_world_rotation() {
  /* return opencv to blender/opengl coordinates transformation matrix */
  // clang-format off
  return torch::tensor({{1, 0, 0}, 
                        {0, 0, -1}, 
                        {0, 1, 0 }}, torch::kFloat32);
  // clang-format on
}

torch::Tensor opencv_to_opengl_camera_rotation() {
  /* return opencv to blender/opengl coordinates transformation matrix */
  // clang-format off
  return torch::tensor({{1, 0, 0}, 
                        {0,-1, 0}, 
                        {0, 0,-1}}, torch::kFloat32);
  // clang-format on
}

torch::Tensor opengl_to_opencv_world_transform() {
  /* return opencv to blender/opengl coordinates transformation matrix */
  // clang-format off
  return torch::tensor({{1, 0, 0, 0}, 
                        {0, 0, -1, 0}, 
                        {0, 1, 0, 0},
                        {0, 0, 0, 1}}, torch::kFloat32);
  // clang-format on
}

torch::Tensor kitti_to_opencv_world_transform() {
  // clang-format off
  return torch::tensor({{ 0, 0, 1, 0}, 
                        {-1, 0, 0, 0}, 
                        { 0,-1, 0, 0},
                        { 0, 0, 0, 1}}, torch::kFloat32);
  // clang-format on
}

torch::Tensor rotpos_to_transform34(const torch::Tensor &rotation,
                                    const torch::Tensor &translation) {
  /* """Converts a rotation matrix and translation vector to a 4x4
  transformation matrix

  Args:
      rotation (torch.Tensor): A 3x3 rotation matrix
      translation (torch.Tensor): A 3x1 translation vector

  Returns:
      torch.Tensor: A 4x4 transformation matrix
  """ */
  return torch::cat({rotation, translation.view({3, 1})}, 1);
}

torch::Tensor rotpos_to_transform44(const torch::Tensor &rotation,
                                    const torch::Tensor &translation) {
  /* """Converts a rotation matrix and translation vector to a 4x4
  transformation matrix

  Args:
      rotation (torch.Tensor): A 3x3 rotation matrix
      translation (torch.Tensor): A 3x1 translation vector

  Returns:
      torch.Tensor: A 4x4 transformation matrix
  """ */
  return torch::cat({rotpos_to_transform34(rotation, translation),
                     torch::tensor({{0, 0, 0, 1}}, torch::kFloat32)});
}

torch::Tensor change_camera_system(const torch::Tensor &pose,
                                   const int &pose_system_type) {
  /**
   * @description: pose_system_type: 0: opencv; 1: blender/opengl
   * @return {*}
   */
  /* clang-format off
    r"""Applies a coordinate system change using the given 3x3 permutation & reflections matrix.

    For instance:

    (1) From a Y-up coordinate system (cartesian) to Z-up:

    .. math::

        \text{basis_change} = \begin{bmatrix}
            1 & 0 & 0 \\
            0 & 0 & -1 \\
            0 & 1 & 0
        \end{bmatrix}

    (2) From a right handed coordinate system (Z pointing outwards) to a left handed one (Z pointing inwards):

    .. math::
        \text{basis_change} = \begin{bmatrix}
            1 & 0 & 0 \\
            0 & 1 & 0 \\
            0 & 0 & -1
        \end{bmatrix}


    The basis_change is assumed to have a determinant of +1 or -1.

    .. seealso::

        :func:`blender_coords()` and :func:`opengl_coords()`

    Args:
        basis_change (numpy.ndarray or torch.Tensor):
            a composition of axes permutation and reflections, of shape :math:`(3, 3)`
    """
    # One prevalent form of performing coordinate change is swapping / negating the inverse view matrix rows.
    # That is - we want to alter the camera axes & position in WORLD coordinates.
    # Note it's enough however, to multiply the R component of the view matrix by the basis change matrix transpose
    # (recall we rotate about the world origin, which remains in place).
    #
    # Compare the inverse matrix before after basis change:
    #               Pre basis change:
    #     view_matrix =             inverse_view_matrix =           Rt is R transposed
    #       [ R | t ]                 [ Rt | -Rt @ t ]              @ denotes matrix column multiplication
    #       [ 0 | 1 ]                 [ 0  |   1     ]
    #
    #               Post basis change:
    #     view_matrix =             inverse_view_matrix =                    P is the basis change matrix
    #       [ R @ Pt | t ]                 [ P @ Rt | -(P @ Rt) @ t ]        Pt is the transposition of P
    #       [ 0      | 1 ]                 [ 0      |       1       ]
    #
    #                                =     [ P @ Rt | P @ (-Rt @ t) ]
    #                                      [ 0      |       1       ]
    clang-format on */
  /* torch::Tensor basis_change = (pose_system_type == 0)
                                   ? opencv_camera_transform()
                                   : blender_camera_transform();

  return pose.matmul(basis_change); */
  switch (pose_system_type) {
  case SystemType::Blender:
    return pose.matmul(opencv_to_blender_camera_transform().to(pose.device()));
  case SystemType::OpenGL:
    return pose.matmul(opencv_to_opengl_world_transform().to(pose.device()));
  case SystemType::Kitti:
    throw std::runtime_error("Invalid system type");
  }
  return {};
}

torch::Tensor change_world_system(const torch::Tensor &pose,
                                  const int &_system_type) {
  /**
   * @description: pose_system_type: 0: opencv; 1: blender; 2: opengl
   * @return {*}
   */
  /* clang-format off
    r"""Applies a coordinate system change using the given 3x3 permutation & reflections matrix.

    For instance:

    (1) From a Y-up coordinate system (cartesian) to Z-up:

    .. math::

        \text{basis_change} = \begin{bmatrix}
            1 & 0 & 0 \\
            0 & 0 & -1 \\
            0 & 1 & 0
        \end{bmatrix}

    (2) From a right handed coordinate system (Z pointing outwards) to a left handed one (Z pointing inwards):

    .. math::
        \text{basis_change} = \begin{bmatrix}
            1 & 0 & 0 \\
            0 & 1 & 0 \\
            0 & 0 & -1
        \end{bmatrix}


    The basis_change is assumed to have a determinant of +1 or -1.

    .. seealso::

        :func:`blender_coords()` and :func:`opengl_coords()`

    Args:
        basis_change (numpy.ndarray or torch.Tensor):
            a composition of axes permutation and reflections, of shape :math:`(3, 3)`
    """
    # One prevalent form of performing coordinate change is swapping / negating the inverse view matrix rows.
    # That is - we want to alter the camera axes & position in WORLD coordinates.
    # Note it's enough however, to multiply the R component of the view matrix by the basis change matrix transpose
    # (recall we rotate about the world origin, which remains in place).
    #
    # Compare the inverse matrix before after basis change:
    #               Pre basis change:
    #     view_matrix =             inverse_view_matrix =           Rt is R transposed
    #       [ R | t ]                 [ Rt | -Rt @ t ]              @ denotes matrix column multiplication
    #       [ 0 | 1 ]                 [ 0  |   1     ]
    #
    #               Post basis change:
    #     view_matrix =             inverse_view_matrix =                    P is the basis change matrix
    #       [ R @ Pt | t ]                 [ P @ Rt | -(P @ Rt) @ t ]        Pt is the transposition of P
    #       [ 0      | 1 ]                 [ 0      |       1       ]
    #
    #                                =     [ P @ Rt | P @ (-Rt @ t) ]
    #                                      [ 0      |       1       ]
    clang-format on */
  torch::Tensor transform_pose;
  if (pose.size(0) == 3) {
    transform_pose = torch::cat(
        {pose,
         torch::tensor({{0, 0, 0, 1}}, torch::kFloat32).to(pose.device())},
        0);
  }

  if (transform_pose.size(-1) == 4) {
    // inicate a transformation
    switch (_system_type) {
    case SystemType::OpenCV:
      return transform_pose.slice(0, 0, pose.size(0));
    case SystemType::Blender:
      return transform_pose.slice(0, 0, pose.size(0));
    case SystemType::OpenGL:
      return opengl_to_opencv_world_transform()
          .to(transform_pose.device())
          .matmul(transform_pose)
          .slice(0, 0, pose.size(0));
    case SystemType::Kitti:
      return kitti_to_opencv_world_transform()
          .to(transform_pose.device())
          .matmul(transform_pose)
          .slice(0, 0, pose.size(0));
    default:
      throw std::runtime_error("Invalid system type");
    }
  } else if (pose.size(-1) == 3) {
  } else {
    throw std::runtime_error("Invalid pose size");
  }
  return {};
}

torch::Tensor reset_world_system(const torch::Tensor &pose,
                                 const int &_system_type) {
  if (pose.size(-1) == 4) {
  } else if (pose.size(-1) == 3) {
    // inicate a translation
    switch (_system_type) {
    case SystemType::OpenGL:
      return pose.matmul(opengl_to_opencv_world_rotation().to(pose.device()));
    default:
      return pose;
    }
  } else {
    throw std::runtime_error("Invalid pose size");
  }
  return {};
}

} // namespace coords