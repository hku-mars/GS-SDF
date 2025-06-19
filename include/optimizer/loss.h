#pragma once

#include <torch/torch.h>

namespace loss {

torch::Tensor gs_sdf_loss(const torch::Tensor &gs_sdf,
                          const torch::Tensor &weigtht,
                          const std::string &name = "");

torch::Tensor gs_sdf_normal_loss(const torch::Tensor &gs_normal,
                                 const torch::Tensor &sdf_normal,
                                 const torch::Tensor &weigtht);

torch::Tensor rgb_loss(const torch::Tensor &rgb, const torch::Tensor &rgb_gt,
                       const torch::Tensor &mask = torch::Tensor());

torch::Tensor distortion_loss(const torch::Tensor &render_dist,
                              const std::string &name = "");

torch::Tensor dssim_loss(const torch::Tensor &pred_image,
                         const torch::Tensor &gt_image,
                         const torch::Tensor &mask = torch::Tensor());

torch::Tensor sdf_loss(const torch::Tensor &pred_sdf,
                       const torch::Tensor &gt_sdf,
                       const torch::Tensor &pred_isigma);

torch::Tensor eikonal_loss(const torch::Tensor &grad,
                           const std::string &name = "");

torch::Tensor curvate_loss(const torch::Tensor &hessian,
                           const std::string &name = "");

torch::Tensor entropy_loss(const torch::Tensor &alpha,
                           const std::string &name = "");

} // namespace loss
