#include "optimizer_utils.h"

using namespace std;

void prune_optimizer(torch::optim::Adam *optimizer, const torch::Tensor &mask,
                     torch::Tensor &old_tensor, int param_position) {
  if (optimizer->param_groups()[param_position].params()[0].defined()) {
    auto tensor_impl = optimizer->param_groups()[param_position]
                           .params()[0]
                           .unsafeGetTensorImpl();
    if (!tensor_impl || !optimizer->state().count(tensor_impl)) {
      return;
    }

    auto state_ptr = optimizer->state()[tensor_impl].get();
    auto adam_state = dynamic_cast<torch::optim::AdamParamState *>(state_ptr);
    if (!adam_state) {
      return;
    }

    auto adamParamStates =
        std::make_unique<torch::optim::AdamParamState>(*adam_state);
    optimizer->state().erase(tensor_impl);

    adamParamStates->exp_avg(adamParamStates->exp_avg().index_select(0, mask));
    adamParamStates->exp_avg_sq(
        adamParamStates->exp_avg_sq().index_select(0, mask));

    optimizer->param_groups()[param_position].params()[0] =
        old_tensor.index_select(0, mask).set_requires_grad(true);
    old_tensor = optimizer->param_groups()[param_position]
                     .params()[0]; // update old tensor
    optimizer->state()[optimizer->param_groups()[param_position]
                           .params()[0]
                           .unsafeGetTensorImpl()] = std::move(adamParamStates);

  } else {
    old_tensor = old_tensor.index_select(0, mask).set_requires_grad(true);
  }
}

void cat_tensors_to_optimizer(torch::optim::Adam *optimizer,
                              const torch::Tensor &extension_tensor,
                              torch::Tensor &old_tensor, int param_position) {
  if (optimizer->param_groups()[param_position].params()[0].defined()) {
    auto tensor_impl = optimizer->param_groups()[param_position]
                           .params()[0]
                           .unsafeGetTensorImpl();
    if (!tensor_impl || !optimizer->state().count(tensor_impl)) {
      return;
    }

    auto state_ptr = optimizer->state()[tensor_impl].get();
    auto adam_state = dynamic_cast<torch::optim::AdamParamState *>(state_ptr);
    if (!adam_state) {
      return;
    }

    auto adamParamStates =
        std::make_unique<torch::optim::AdamParamState>(*adam_state);
    optimizer->state().erase(tensor_impl);

    adamParamStates->exp_avg(torch::cat(
        {adamParamStates->exp_avg(), torch::zeros_like(extension_tensor)}, 0));
    adamParamStates->exp_avg_sq(torch::cat(
        {adamParamStates->exp_avg_sq(), torch::zeros_like(extension_tensor)},
        0));

    optimizer->param_groups()[param_position].params()[0] =
        torch::cat({old_tensor, extension_tensor}, 0).set_requires_grad(true);
    old_tensor = optimizer->param_groups()[param_position].params()[0];

    optimizer->state()[optimizer->param_groups()[param_position]
                           .params()[0]
                           .unsafeGetTensorImpl()] = std::move(adamParamStates);

  } else {
    old_tensor =
        torch::cat({old_tensor, extension_tensor}, 0).set_requires_grad(true);
  }
}

void prune_cat_tensors_to_optimizer(torch::optim::Adam *optimizer,
                                    torch::Tensor &old_tensor,
                                    const torch::Tensor &rest_idx,
                                    const torch::Tensor &extension_tensor,
                                    int param_position) {
  if (optimizer->param_groups()[param_position].params()[0].defined()) {
    auto tensor_impl = optimizer->param_groups()[param_position]
                           .params()[0]
                           .unsafeGetTensorImpl();
    if (!tensor_impl || !optimizer->state().count(tensor_impl)) {
      return;
    }

    auto state_ptr = optimizer->state()[tensor_impl].get();
    auto adam_state = dynamic_cast<torch::optim::AdamParamState *>(state_ptr);
    if (!adam_state) {
      return;
    }

    auto adamParamStates =
        std::make_unique<torch::optim::AdamParamState>(*adam_state);
    optimizer->state().erase(tensor_impl);

    adamParamStates->exp_avg(
        torch::cat({adamParamStates->exp_avg().index_select(0, rest_idx),
                    torch::zeros_like(extension_tensor)},
                   0));
    adamParamStates->exp_avg_sq(
        torch::cat({adamParamStates->exp_avg_sq().index_select(0, rest_idx),
                    torch::zeros_like(extension_tensor)},
                   0));

    optimizer->param_groups()[param_position].params()[0] =
        torch::cat({old_tensor.index_select(0, rest_idx), extension_tensor}, 0)
            .set_requires_grad(true);
    old_tensor = optimizer->param_groups()[param_position].params()[0];

    optimizer->state()[optimizer->param_groups()[param_position]
                           .params()[0]
                           .unsafeGetTensorImpl()] = std::move(adamParamStates);
  } else {
    old_tensor =
        torch::cat({old_tensor.index_select(0, rest_idx), extension_tensor}, 0)
            .set_requires_grad(true);
  }
}

void replace_tensors_to_optimizer(torch::optim::Adam *optimizer,
                                  torch::Tensor &old_tensor,
                                  torch::Tensor &new_tensor,
                                  int param_position) {
  if (optimizer->param_groups()[param_position].params()[0].defined()) {
    auto tensor_impl = optimizer->param_groups()[param_position]
                           .params()[0]
                           .unsafeGetTensorImpl();
    if (!tensor_impl || !optimizer->state().count(tensor_impl)) {
      return;
    }

    auto state_ptr = optimizer->state()[tensor_impl].get();
    auto adam_state = dynamic_cast<torch::optim::AdamParamState *>(state_ptr);
    if (!adam_state) {
      return;
    }

    auto adamParamStates =
        std::make_unique<torch::optim::AdamParamState>(*adam_state);
    optimizer->state().erase(tensor_impl);

    adamParamStates->exp_avg(torch::zeros_like(new_tensor));
    adamParamStates->exp_avg_sq(torch::zeros_like(new_tensor));

    optimizer->param_groups()[param_position].params()[0] =
        new_tensor.set_requires_grad(true);
    old_tensor = optimizer->param_groups()[param_position].params()[0];

    optimizer->state()[optimizer->param_groups()[param_position]
                           .params()[0]
                           .unsafeGetTensorImpl()] = std::move(adamParamStates);
  } else {
    old_tensor = new_tensor.set_requires_grad(true);
  }
}
