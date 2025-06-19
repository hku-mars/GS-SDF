#pragma once

#include <torch/torch.h>

namespace loss_utils {
torch::Tensor gaussian(int window_size, float sigma) {
  torch::Tensor gauss = torch::empty(window_size);
  for (int x = 0; x < window_size; x++) {
    gauss[x] = exp(-pow(x - window_size / 2, 2) / (2 * pow(sigma, 2)));
  }
  return gauss / gauss.sum();
}

torch::Tensor create_window(int window_size, int channel) {
  torch::Tensor _1D_window = gaussian(window_size, 1.5).unsqueeze(1);
  torch::Tensor _2D_window = _1D_window.mm(_1D_window.t())
                                 .to(torch::kFloat32)
                                 .unsqueeze(0)
                                 .unsqueeze(0);
  torch::Tensor window = torch::autograd::Variable(
      _2D_window.expand({channel, 1, window_size, window_size}).contiguous());
  return window;
}

torch::Tensor _ssim(const torch::Tensor &img1, const torch::Tensor &img2,
                    const torch::Tensor &window, int window_size, int channel,
                    bool size_average = true) {
  torch::Tensor mu1 =
      torch::conv2d(img1, window, {}, 1, window_size / 2, 1, channel);
  torch::Tensor mu2 =
      torch::conv2d(img2, window, {}, 1, window_size / 2, 1, channel);

  torch::Tensor mu1_sq = mu1.pow(2);
  torch::Tensor mu2_sq = mu2.pow(2);
  torch::Tensor mu1_mu2 = mu1 * mu2;

  torch::Tensor sigma1_sq =
      torch::conv2d(img1 * img1, window, {}, 1, window_size / 2, 1, channel) -
      mu1_sq;
  torch::Tensor sigma2_sq =
      torch::conv2d(img2 * img2, window, {}, 1, window_size / 2, 1, channel) -
      mu2_sq;
  torch::Tensor sigma12 =
      torch::conv2d(img1 * img2, window, {}, 1, window_size / 2, 1, channel) -
      mu1_mu2;

  static float C1 = 0.01 * 0.01;
  static float C2 = 0.03 * 0.03;

  torch::Tensor ssim_map =
      ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) /
      ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2));

  if (size_average) {
    return ssim_map.mean();
  } else {
    return ssim_map.mean(1).mean(1).mean(1);
  }
}

torch::Tensor ssim(const torch::Tensor &img1, const torch::Tensor &img2,
                   int window_size = 11, bool size_average = true) {
  // [N, chaneel, H, W], data range: [0, 1]
  int channel = img1.size(-3);
  torch::Tensor window = create_window(window_size, channel);

  if (img1.is_cuda()) {
    window = window.cuda();
  }
  window = window.to(img1.dtype());

  return _ssim(img1, img2, window, window_size, channel, size_average);
}

} // namespace loss_utils