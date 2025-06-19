#include "loss.h"

#include "loss_utils/loss_utils.h"

namespace loss {

torch::Tensor gs_sdf_loss(const torch::Tensor &gs_sdf,
                          const torch::Tensor &weigtht,
                          const std::string &name) {
  return 0.5f * (weigtht * gs_sdf.square()).sum();
}

torch::Tensor gs_sdf_normal_loss(const torch::Tensor &gs_normal,
                                 const torch::Tensor &sdf_normal,
                                 const torch::Tensor &weigtht) {
  return (weigtht * (1.0f - (gs_normal * sdf_normal).sum(-1, true).abs())
                        .abs()
                        .nan_to_num())
      .sum();
}

torch::Tensor rgb_loss(const torch::Tensor &rgb, const torch::Tensor &rgb_gt,
                       const torch::Tensor &mask) {
  if (mask.defined()) {
    return torch::abs((rgb - rgb_gt) * mask).mean();
  }

  // return torch::sqrt((rgb - rgb_gt).square() + 1e-4f).mean();
  return torch::abs(rgb - rgb_gt).mean();
}

torch::Tensor distortion_loss(const torch::Tensor &render_dist,
                              const std::string &name) {
  return render_dist.square().mean();
}

torch::Tensor dssim_loss(const torch::Tensor &pred_image,
                         const torch::Tensor &gt_image,
                         const torch::Tensor &mask) {
  if (mask.defined()) {
    return 1.0f -
           loss_utils::ssim((pred_image * mask).permute({2, 0, 1}).unsqueeze(0),
                            (gt_image * mask).permute({2, 0, 1}).unsqueeze(0));
  }
  return 1.0f - loss_utils::ssim(pred_image.permute({2, 0, 1}).unsqueeze(0),
                                 gt_image.permute({2, 0, 1}).unsqueeze(0));
}

torch::Tensor sdf_loss(const torch::Tensor &pred_sdf,
                       const torch::Tensor &gt_sdf,
                       const torch::Tensor &pred_isigma) {
  auto isigma = pred_isigma.clamp_max(5e2f);

  // auto pred_clamp_mask =
  //     ((-pred_sdf * isigma) < -15.0f) | ((-pred_sdf * isigma) > 15.0f);
  // auto gt_clamp_mask =
  //     (gt_sdf * isigma < 1e-7f) | (gt_sdf * isigma > (1 - 1e-7f));
  // auto clamp_mask = pred_clamp_mask | gt_clamp_mask;
  // std::cout << "pred_clamp_mask/all_number: "
  //           << pred_clamp_mask.sum().item<int>() << " / "
  //           << pred_clamp_mask.numel() << std::endl;
  // std::cout << "gt_clamp_mask/all_number: " <<
  // gt_clamp_mask.sum().item<int>()
  //           << " / " << gt_clamp_mask.numel() << std::endl;
  // std::cout << "clamp_numer/all_number: " << clamp_mask.sum().item<int>()
  //           << " / " << clamp_mask.numel() << std::endl;
  // better avoid nan
  // torch::Tensor sdf_loss = torch::binary_cross_entropy_with_logits(
  //     (-pred_sdf * isigma).clamp(-15.0f, 15.0f),
  //     torch::sigmoid(-gt_sdf * isigma).clamp(1e-7f, 1 - 1e-7f));
  torch::Tensor sdf_loss = torch::binary_cross_entropy_with_logits(
      -pred_sdf * isigma,
      torch::sigmoid(-gt_sdf * isigma).clamp(1e-7f, 1 - 1e-7f));
  // // better avoid nan
  // torch::Tensor sdf_loss = torch::binary_cross_entropy_with_logits(
  //     -pred_sdf * isigma, torch::sigmoid(-gt_sdf * isigma));

  return sdf_loss;
}

torch::Tensor eikonal_loss(const torch::Tensor &grad, const std::string &name) {
  return (grad.norm(2, 1) - 1.0f).square().mean();
}

torch::Tensor curvate_loss(const torch::Tensor &hessian,
                           const std::string &name) {
  auto curvate_loss = hessian.sum(-1).abs().mean();
  curvate_loss.nan_to_num_(0.0f, 0.0f, 0.0f);
  return curvate_loss;
}

} // namespace loss
