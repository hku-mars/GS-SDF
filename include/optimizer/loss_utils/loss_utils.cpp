#include "loss_utils.h"
#include "opencv2/opencv.hpp"

namespace loss_utils {
// 1D Gaussian kernel
torch::Tensor gaussian(int window_size, float sigma) {
  torch::Tensor gauss = torch::empty(window_size);
  for (int x = 0; x < window_size; ++x) {
    gauss[x] = std::exp(
        -(std::pow(std::floor(static_cast<float>(x - window_size) / 2.f), 2)) /
        (2.f * sigma * sigma));
  }
  return gauss / gauss.sum();
}

torch::Tensor create_window(int window_size, int channel) {
  auto _1D_window = gaussian(window_size, 1.5).unsqueeze(1);
  auto _2D_window = _1D_window.mm(_1D_window.t()).unsqueeze(0).unsqueeze(0);
  return _2D_window.expand({channel, 1, window_size, window_size}).contiguous();
}

torch::Tensor _ssim(const torch::Tensor &img1, const torch::Tensor &img2,
                    const torch::Tensor &window, int window_size, int channel,
                    bool size_average) {
  auto mu1 =
      torch::nn::functional::conv2d(img1, window,
                                    torch::nn::functional::Conv2dFuncOptions()
                                        .padding(window_size / 2)
                                        .groups(channel));
  auto mu1_sq = mu1.pow(2);
  auto sigma1_sq =
      torch::nn::functional::conv2d(img1 * img1, window,
                                    torch::nn::functional::Conv2dFuncOptions()
                                        .padding(window_size / 2)
                                        .groups(channel)) -
      mu1_sq;

  auto mu2 =
      torch::nn::functional::conv2d(img2, window,
                                    torch::nn::functional::Conv2dFuncOptions()
                                        .padding(window_size / 2)
                                        .groups(channel));
  auto mu2_sq = mu2.pow(2);
  auto sigma2_sq =
      torch::nn::functional::conv2d(img2 * img2, window,
                                    torch::nn::functional::Conv2dFuncOptions()
                                        .padding(window_size / 2)
                                        .groups(channel)) -
      mu2_sq;

  auto mu1_mu2 = mu1 * mu2;
  auto sigma12 =
      torch::nn::functional::conv2d(img1 * img2, window,
                                    torch::nn::functional::Conv2dFuncOptions()
                                        .padding(window_size / 2)
                                        .groups(channel)) -
      mu1_mu2;

  static const float C1 = 0.01 * 0.01;
  static const float C2 = 0.03 * 0.03;
  auto ssim_map = ((2.f * mu1_mu2 + C1) * (2.f * sigma12 + C2)) /
                  ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2));

  if (size_average) {
    return ssim_map.mean();
  } else {
    return ssim_map.mean(1).mean(1).mean(1);
  }
}

torch::Tensor ssim(const torch::Tensor &img1, const torch::Tensor &img2,
                   int window_size, int channel) {
  static const float C1 = 0.01 * 0.01;
  static const float C2 = 0.03 * 0.03;
  const auto window = create_window(window_size, channel)
                          .to(torch::kFloat32)
                          .to(torch::kCUDA, true);
  // [N, chaneel, H, W], data range: [0, 1]
  auto mu1 =
      torch::nn::functional::conv2d(img1, window,
                                    torch::nn::functional::Conv2dFuncOptions()
                                        .padding(window_size / 2)
                                        .groups(channel));
  auto mu1_sq = mu1.pow(2);
  auto sigma1_sq =
      torch::nn::functional::conv2d(img1 * img1, window,
                                    torch::nn::functional::Conv2dFuncOptions()
                                        .padding(window_size / 2)
                                        .groups(channel)) -
      mu1_sq;

  auto mu2 =
      torch::nn::functional::conv2d(img2, window,
                                    torch::nn::functional::Conv2dFuncOptions()
                                        .padding(window_size / 2)
                                        .groups(channel));
  auto mu2_sq = mu2.pow(2);
  auto sigma2_sq =
      torch::nn::functional::conv2d(img2 * img2, window,
                                    torch::nn::functional::Conv2dFuncOptions()
                                        .padding(window_size / 2)
                                        .groups(channel)) -
      mu2_sq;

  auto mu1_mu2 = mu1 * mu2;
  auto sigma12 =
      torch::nn::functional::conv2d(img1 * img2, window,
                                    torch::nn::functional::Conv2dFuncOptions()
                                        .padding(window_size / 2)
                                        .groups(channel)) -
      mu1_mu2;

  auto ssim_map = ((2.f * mu1_mu2 + C1) * (2.f * sigma12 + C2)) /
                  ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2));

  return ssim_map.mean();
}

float psnr(const torch::Tensor &rendered_img, const torch::Tensor &gt_img) {
  // Check both are in shape of [1, chaneel, H, W]
  TORCH_CHECK(
      rendered_img.dim() == 4 && gt_img.dim() == 4,
      "Both rendered_img and gt_img should be in shape of [1, chaneel, H, W]");
  torch::Tensor squared_diff =
      (rendered_img - gt_img.to(rendered_img.device())).square();
  torch::Tensor mse_val =
      squared_diff.reshape({rendered_img.size(0), -1}).mean(1, true);
  return (20.f * torch::log10(1.0f / mse_val.sqrt())).mean().item<float>();
}

} // namespace loss_utils