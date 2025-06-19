#pragma once

#include <torch/torch.h>

namespace loss_utils {
torch::Tensor gaussian(int window_size, float sigma);

torch::Tensor create_window(int window_size, int channel);

torch::Tensor _ssim(const torch::Tensor &img1, const torch::Tensor &img2,
                    const torch::Tensor &window, int window_size, int channel,
                    bool size_average = true);

// torch::Tensor ssim(const torch::Tensor &img1, const torch::Tensor &img2,
//                    int window_size = 11, bool size_average = true);
torch::Tensor ssim(const torch::Tensor &img1, const torch::Tensor &img2,
                   int window_size = 11, int channel = 3);
float psnr(const torch::Tensor &rendered_img, const torch::Tensor &gt_img);

} // namespace loss_utils