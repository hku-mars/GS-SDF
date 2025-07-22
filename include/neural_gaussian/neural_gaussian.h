#pragma once

#include "neural_net/local_map.h"
#include <torch/torch.h>

struct NeuralGS : torch::nn::Module {
  typedef std::shared_ptr<NeuralGS> Ptr;
  NeuralGS(const LocalMap::Ptr &_local_map_ptr, const torch::Tensor &_points,
           const int &_num_train_data, const float &_spatial_scale,
           const bool &_sdf_enable = false);
  NeuralGS(const LocalMap::Ptr &_local_map_ptr,
           const std::filesystem::path &input_path);

  LocalMap::Ptr local_map_ptr_;
  float spatial_scale_, original_spatial_scale_;
  bool sdf_enable_;
  std::vector<torch::optim::OptimizerParamGroup> optimizer_params_groups_;
  int gs_param_start_idx; // gs_param_start_idx_in_optimier
  // neural gaussian parameter
  torch::Tensor anchors_; // [N, 3]

  torch::Tensor offsets_;    // [N, K, 3]
  torch::Tensor scaling_;    // [N, K, 3]
  torch::Tensor quaternion_; // [N, K, 4]
  torch::Tensor opacity_;    // [N, K, 1]
  at::optional<int> sh_degree_to_use_;
  torch::Tensor features_dc_;   // [N, 1, 3]
  torch::Tensor features_rest_; // [N, D-1, 3]

  // strategy config
  std::string key_for_gradient;
  int pause_refine_after_reset;

  int num_train_data_;

  // lock for render
  std::mutex render_mutex_;

  torch::Tensor get_xyz();
  torch::Tensor get_scale();
  torch::Tensor get_opacity(const bool &training = false);

  std::map<std::string, torch::Tensor>
  render(const torch::Tensor &_pose_cam2world, sensor::Cameras camera,
         const bool &training = false, const int &bck_color = 0);

  std::map<std::string, torch::Tensor> state;
  void train_callback(const int &iter, const int &_total_iter,
                      const std::shared_ptr<torch::optim::Adam> &_p_optimizer,
                      std::map<std::string, torch::Tensor> &info);

  void export_gs_to_ply(std::filesystem::path &output_path,
                        const bool &export_as_3dgs = false);
  void load_ply_to_gs(const std::filesystem::path &input_path);

  void freeze_structure();
  void unfreeze_structure();

private:
  void update_state(std::map<std::string, torch::Tensor> &info);
  void zero_state();
  void grow_gs(const int &iter,
               const std::shared_ptr<torch::optim::Adam> &_p_optimizer);
  int duplicate(const std::shared_ptr<torch::optim::Adam> &_p_optimizer,
                const torch::Tensor &is_dupli);
  int split(const std::shared_ptr<torch::optim::Adam> &_p_optimizer,
            const torch::Tensor &is_split);
  int prune_gs(const std::shared_ptr<torch::optim::Adam> &_p_optimizer,
               const torch::Tensor &is_prune);
  void prune_gs(const int &iter,
                const std::shared_ptr<torch::optim::Adam> &_p_optimizer,
                const bool &prune_opa_only = false);
  void
  prune_invisible_gs(const int &iter,
                     const std::shared_ptr<torch::optim::Adam> &_p_optimizer,
                     const bool &prune_opa_only = false);
  void prune_nan_gs(const int &iter,
                    const std::shared_ptr<torch::optim::Adam> &_p_optimizer);
  void reset_opacity(const std::shared_ptr<torch::optim::Adam> &_p_optimizer);

  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
             torch::Tensor, torch::Tensor>
  generate_gaussian(const bool &training = false,
                    const torch::Tensor &view_pos = torch::Tensor(),
                    const torch::Tensor &mask = torch::Tensor());
};