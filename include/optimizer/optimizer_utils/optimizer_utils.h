#pragma once
#include <torch/torch.h>

void prune_optimizer(torch::optim::Adam *optimizer, const torch::Tensor &mask,
                     torch::Tensor &old_tensor, int param_position);

void cat_tensors_to_optimizer(torch::optim::Adam *optimizer,
                              const torch::Tensor &extension_tensor,
                              torch::Tensor &old_tensor, int param_position);

void prune_cat_tensors_to_optimizer(torch::optim::Adam *optimizer,
                                    torch::Tensor &old_tensor,
                                    const torch::Tensor &rest_idx,
                                    const torch::Tensor &extension_tensor,
                                    int param_position);

void replace_tensors_to_optimizer(torch::optim::Adam *optimizer,
                                  torch::Tensor &old_tensor,
                                  torch::Tensor &new_tensor,
                                  int param_position);
