#include "neural_gaussian.h"

#include "gsplat_cpp/fully_fused_projection.h"
#include "gsplat_cpp/rasterize_to_pixels.h"
#include "llog/llog.h"

#include "gsplat_cpp/rendering.h"
#include "optimizer/optimizer_utils/optimizer_utils.h"
#include "params/params.h"

#include "gauss_utils.hpp"
#include "utils/ply_utils/ply_utils_torch.h"

using namespace std;

std::map<std::string, torch::Tensor>
init_gs_with_sdf(const LocalMap::Ptr &_local_map_ptr, const torch::Tensor &xyzs,
                 const float mesh_res, const bool &init_opa = false) {
  torch::NoGradGuard no_grad;

  // Define batch size to avoid OOM
  const int batch_size = k_vis_batch_pt_num; // Adjust based on your GPU memory
  const int num_points = xyzs.size(0);
  const int num_batches =
      (num_points + batch_size - 1) / batch_size; // Ceiling division

  // Create tensors to store the results
  torch::Tensor all_grad, all_curv_dom, all_quaternion, all_opacity;
  auto device = xyzs.device();
  auto options = xyzs.options();

  all_grad = torch::empty({num_points, 3}, options);
  all_curv_dom = torch::empty({num_points, 3}, options);
  all_quaternion = torch::empty({num_points, 4}, options);

  if (init_opa) {
    all_opacity = torch::empty({num_points}, options);
  }

  // Process each batch
  for (int i = 0; i < num_batches; i++) {
    int start_idx = i * batch_size;
    int end_idx = std::min((i + 1) * batch_size, num_points);
    int current_batch_size = end_idx - start_idx;

    // Get the batch of points
    auto batch_xyzs = xyzs.slice(0, start_idx, end_idx);

    // Process the batch
    std::vector<torch::Tensor> grad_results =
        _local_map_ptr->get_gradient(batch_xyzs, mesh_res, {}, true);

    auto grad = grad_results[0];
    auto curv_dom = grad_results[1];

    // Normalize gradients
    auto c = torch::nn::functional::normalize(
        grad, torch::nn::functional::NormalizeFuncOptions().dim(-1));

    auto tmp_basis = torch::nn::functional::normalize(
        curv_dom, torch::nn::functional::NormalizeFuncOptions().dim(-1));

    // [N,3,3]
    auto rot = utils::rotation_6d_to_matrix(torch::cat({c, tmp_basis}, -1));
    rot = torch::stack(
        {rot.select(-1, 1), rot.select(-1, 2), rot.select(-1, 0)}, -1);

    // Convert rotation to quaternion
    // rot2axis-angle2quat seems easier than rot2quat
    // https://zhuanlan.zhihu.com/p/691639068 : rot2axis-angle
    // https://zhuanlan.zhihu.com/p/698061865 : axis-angle2quat
    // [N]
    auto rot_trace = rot.index({torch::indexing::Slice(), 0, 0}) +
                     rot.index({torch::indexing::Slice(), 1, 1}) +
                     rot.index({torch::indexing::Slice(), 2, 2});

    // [N,1]
    auto angle = torch::acos((rot_trace.unsqueeze(-1) - 1.0f) * 0.5f);

    // [N,3]
    auto axis = torch::stack({rot.index({torch::indexing::Slice(), 2, 1}) -
                                  rot.index({torch::indexing::Slice(), 1, 2}),
                              rot.index({torch::indexing::Slice(), 0, 2}) -
                                  rot.index({torch::indexing::Slice(), 2, 0}),
                              rot.index({torch::indexing::Slice(), 1, 0}) -
                                  rot.index({torch::indexing::Slice(), 0, 1})},
                             -1) /
                (2.0f * torch::sin(angle));

    axis = torch::nn::functional::normalize(
        axis, torch::nn::functional::NormalizeFuncOptions().dim(-1));

    auto batch_quaternion = torch::cat(
        {torch::cos(angle * 0.5f), torch::sin(angle * 0.5f) * axis}, -1);

    // Handle NaN values
    batch_quaternion = batch_quaternion.nan_to_num();

    // Store results in the output tensors
    all_grad.slice(0, start_idx, end_idx).copy_(grad);
    all_curv_dom.slice(0, start_idx, end_idx).copy_(curv_dom);
    all_quaternion.slice(0, start_idx, end_idx).copy_(batch_quaternion);

    if (init_opa) {
      // Calculate opacity for this batch
      auto sdf_results = _local_map_ptr->get_sdf(batch_xyzs);
      auto sdf = sdf_results[0];
      auto isigma = sdf_results[1];
      auto batch_opacity = torch::exp(-sdf.square() * isigma).squeeze();

      all_opacity.slice(0, start_idx, end_idx).copy_(batch_opacity);
    }
  }

  if (init_opa) {
    return {{"quaternion", all_quaternion},
            {"opacity", all_opacity},
            {"grad", all_grad},
            {"curv_dom", all_curv_dom}};
  }
  return {{"quaternion", all_quaternion},
          {"grad", all_grad},
          {"curv_dom", all_curv_dom}};
}

std::tuple<torch::Tensor, torch::Tensor, std::map<std::string, torch::Tensor>>
rasterization_2dgs_sdf(
    const torch::Tensor &means,     //[N, 3]
    const torch::Tensor &quats,     // [N, 4]
    const torch::Tensor &scales,    // [N, 3]
    const torch::Tensor &opacities, // [N]
    const torch::Tensor &colors,    //[(C,) N, D] or [(C,) N, K, 3]
    const torch::Tensor &viewmats,  //[C, 4, 4]
    const torch::Tensor &Ks,        //[C, 3, 3]
    int width, int height, const std::string &render_mode, float near_plane,
    float far_plane, float radius_clip, at::optional<int> sh_degree,
    bool packed, int tile_size, at::optional<torch::Tensor> backgrounds,
    bool sparse_grad, bool absgrad, bool distloss,
    const LocalMap::Ptr &_local_map_ptr, const bool &training) {
  static auto p_t_pre = llog::CreateTimer("pre");
  p_t_pre->tic();

  std::map<std::string, torch::Tensor> meta;

  auto N = means.size(0);
  auto C = viewmats.size(0);
  auto device = means.device();

  TORCH_CHECK(means.sizes() == torch::IntArrayRef({N, 3}),
              "Invalid means shape");
  TORCH_CHECK(quats.sizes() == torch::IntArrayRef({N, 4}),
              "Invalid quats shape");
  TORCH_CHECK(scales.sizes() == torch::IntArrayRef({N, 3}),
              "Invalid scales shape");
  TORCH_CHECK(opacities.sizes() == torch::IntArrayRef({N}),
              "Invalid opacities shape", opacities.sizes(),
              torch::IntArrayRef({N}));
  TORCH_CHECK(viewmats.sizes() == torch::IntArrayRef({C, 4, 4}),
              "Invalid viewmats shape");
  TORCH_CHECK(Ks.sizes() == torch::IntArrayRef({C, 3, 3}), "Invalid Ks shape");
  TORCH_CHECK(render_mode == "RGB" || render_mode == "D" ||
                  render_mode == "ED" || render_mode == "RGB+D" ||
                  render_mode == "RGB+ED",
              "Invalid render_mode");

  if (sh_degree.has_value()) {
    // # treat colors as SH coefficients, should be in shape [N, K, 3] or [C, N,
    // K, 3] # Allowing for activating partial SH bands
    TORCH_CHECK(
        (colors.dim() == 3 && colors.size(0) == N && colors.size(2) == 3),
        "Invalid colors shape");
    TORCH_CHECK((std::pow(sh_degree.value() + 1, 2) <= colors.size(1)),
                "Invalid colors shape");
  } else {
    TORCH_CHECK((colors.dim() == 2 && colors.size(0) == N) ||
                    (colors.dim() == 3 &&
                     colors.sizes().slice(0, 2) == torch::IntArrayRef({C, N})),
                "Invalid colors shape");
  }
  p_t_pre->toc_sum();
  static auto p_t_proj = llog::CreateTimer("proj");
  p_t_proj->tic();

  // # Compute Ray-Splat intersection transformation.
  auto [camera_ids, gaussian_ids, radii, means2d, depths, ray_transforms,
        normals, samples, samples_weights] =
      fully_fused_projection_2dgs(means, quats, scales, viewmats, Ks, width,
                                  height, near_plane, far_plane, radius_clip,
                                  packed, sparse_grad);
  auto pt_opacities = opacities.index({gaussian_ids});

  p_t_proj->toc_sum();
  static auto p_t_sh = llog::CreateTimer("sh");
  p_t_sh->tic();

  torch::Tensor pt_colors = gsplat_cpp::get_view_colors(
      viewmats, means, radii, colors, camera_ids, gaussian_ids, sh_degree);

  p_t_sh->toc_sum();
  static auto p_t_tile = llog::CreateTimer("tile");
  p_t_tile->tic();

  // TODO: depth should not be calculated by center.z
  auto [tiles_per_gauss, flatten_ids, isect_offsets] =
      gsplat_cpp::tile_encode(width, height, tile_size, means2d, radii, depths,
                              packed, C, camera_ids, gaussian_ids);

  p_t_tile->toc_sum();
  static auto p_t_ras = llog::CreateTimer("ras");
  p_t_ras->tic();

  auto means2d_absgrad = torch::zeros_like(means2d).requires_grad_(absgrad);
  auto densify =
      torch::zeros_like(means2d, means.options().requires_grad(true));
  auto [render_colors, render_depths, render_alphas, render_normals,
        render_distort, render_median, visibilities] =
      rasterize_to_pixels_2dgs(means2d, ray_transforms, pt_colors, pt_opacities,
                               normals, densify, width, height, tile_size,
                               isect_offsets, flatten_ids, backgrounds,
                               at::nullopt, packed, means2d_absgrad, distloss);

  if (absgrad) {
    meta["absgrad"] = means2d_absgrad;
  }

  if (render_mode == "ED" || render_mode == "RGB+ED") {
    render_depths = (render_depths / render_alphas).nan_to_num();
  }

  render_colors = torch::cat({render_colors, render_depths}, -1);

  // transform normal to world space
  render_normals =
      render_normals.matmul(viewmats.inverse()
                                .index({0, torch::indexing::Slice(0, 3),
                                        torch::indexing::Slice(0, 3)})
                                .t());

  p_t_ras->toc_sum();
  static auto p_t_meta = llog::CreateTimer("meta");
  p_t_meta->tic();

  meta["render_normal"] = render_normals;
  meta["render_median"] = render_median;

  // gs normal in camera space
  meta["normal"] = normals;
  meta["gaussian_ids"] = gaussian_ids;
  meta["radii"] = radii;
  meta["gradient_2dgs"] = densify;

  meta["width"] = torch::tensor({width});
  meta["height"] = torch::tensor({height});
  meta["n_cameras"] = torch::tensor({C});

  if (k_center_reg) {
    meta["samples"] = means.index_select(0, gaussian_ids);
    meta["samples_weights"] = torch::ones_like(samples_weights);
  } else {
    meta["samples"] = samples;
    meta["samples_weights"] = samples_weights;
  }
  meta["samples_opacities"] = pt_opacities;
  meta["visibilities"] = visibilities;

  p_t_meta->toc_sum();
  return std::make_tuple(render_colors, render_alphas, meta);
}

NeuralGS::NeuralGS(const LocalMap::Ptr &_local_map_ptr,
                   const torch::Tensor &_points, const int &_num_train_data,
                   const float &_spatial_scale, const bool &_sdf_enable)
    : local_map_ptr_(_local_map_ptr), num_train_data_(_num_train_data),
      original_spatial_scale_(_spatial_scale), sdf_enable_(_sdf_enable),
      torch::nn::Module() {
  torch::NoGradGuard no_grad;

  // avoid too large pos lr degenerate the training
  spatial_scale_ = std::min(original_spatial_scale_, 2.f);
  k_grow_scale3d =
      std::min(k_grow_scale3d * original_spatial_scale_, 0.25f * k_leaf_size);

  key_for_gradient = "gradient_2dgs";

  if (k_pause_refine) {
    pause_refine_after_reset = _num_train_data;
  } else {
    pause_refine_after_reset = 0;
  }
  auto device = _points.device();

  float mesh_res = 0.5f * k_leaf_size;
  local_map_ptr_->meshing_(mesh_res, true);
  local_map_ptr_->p_mesher_->save_mesh(k_output_path, k_vis_attribute, "gs_",
                                       true);

  auto valid_vertices_idx = get<0>(torch::unique_dim(
      local_map_ptr_->p_mesher_->faces_.to(k_device).view({-1}), 0));

  if (valid_vertices_idx.size(0) > k_vis_batch_pt_num) {
    int sample_step = valid_vertices_idx.size(0) / k_vis_batch_pt_num;
    sample_step = std::max(sample_step, 1);
    valid_vertices_idx = valid_vertices_idx.slice(0, 0, -1, sample_step);
  }
  anchors_ = local_map_ptr_->p_mesher_->vertices_.to(k_device).index_select(
      0, valid_vertices_idx);

  scaling_ =
      torch::full({anchors_.size(0), 3}, log(mesh_res), anchors_.options())
          .to(device);

  auto num_points = anchors_.size(0);
  std::map<std::string, torch::Tensor> sdf_init_gs_results;
  if (sdf_enable_ && k_geo_init) {
    sdf_init_gs_results =
        init_gs_with_sdf(local_map_ptr_, anchors_, mesh_res, true);
    quaternion_ = sdf_init_gs_results["quaternion"];
    opacity_ = sdf_init_gs_results["opacity"];
  } else {
    quaternion_ = random_quat_tensor(num_points).to(device);
    // [N, K, 1]
    opacity_ = torch::logit(0.1f * torch::ones({num_points}, device));
  }

  // sky initialization: uniformly generate Gaussian on a sphere
  if (k_sky_init) {
    int num_sky_points =
        1000 *
        original_spatial_scale_; // Number of points to generate on the sphere

    // 生成三维高斯随机数 [N,3]
    auto sphere_samples = torch::randn(
        {num_sky_points, 3}, torch::dtype(torch::kFloat).device(device));
    sphere_samples = torch::nn::functional::normalize(
        sphere_samples,
        torch::nn::functional::NormalizeFuncOptions().dim(-1).eps(1e-6));
    // 归一化得到单位球面点
    auto sphere_radius = 0.6f * k_inner_map_size;
    k_far = 2.0f * sphere_radius;
    auto sky_anchor = sphere_samples * sphere_radius +
                      k_map_origin.view({1, 3}).to(sphere_samples.device());

    // Size proportional to sphere radius
    // 4πr^2 / num_sky_points * 1.1 (inflation factor)
    auto sky_scale = torch::full(
        {num_sky_points, 3},
        log(1.1f * M_PI * sphere_radius * sphere_radius / num_sky_points),
        device);
    auto sky_tmp_basis = torch::stack({sphere_samples.select(-1, 1),
                                       sphere_samples.select(-1, 2),
                                       sphere_samples.select(-1, 0)},
                                      -1);
    auto rot = utils::rotation_6d_to_matrix(
        torch::cat({sphere_samples, sky_tmp_basis}, -1));
    rot = torch::stack(
        {rot.select(-1, 1), rot.select(-1, 2), rot.select(-1, 0)}, -1);

    // [N]
    auto rot_trace = rot.index({torch::indexing::Slice(), 0, 0}) +
                     rot.index({torch::indexing::Slice(), 1, 1}) +
                     rot.index({torch::indexing::Slice(), 2, 2});
    // [N,1]
    auto angle = torch::acos((rot_trace.unsqueeze(-1) - 1.0f) * 0.5f);
    // [N,3]
    auto axis = torch::stack({rot.index({torch::indexing::Slice(), 2, 1}) -
                                  rot.index({torch::indexing::Slice(), 1, 2}),
                              rot.index({torch::indexing::Slice(), 0, 2}) -
                                  rot.index({torch::indexing::Slice(), 2, 0}),
                              rot.index({torch::indexing::Slice(), 1, 0}) -
                                  rot.index({torch::indexing::Slice(), 0, 1})},
                             -1) /
                (2.0f * torch::sin(angle));
    axis = torch::nn::functional::normalize(
        axis, torch::nn::functional::NormalizeFuncOptions().dim(-1));
    auto sky_quat = torch::cat(
        {torch::cos(angle * 0.5f), torch::sin(angle * 0.5f) * axis}, -1);
    // tackle angle = 0 leads to nan
    sky_quat = sky_quat.nan_to_num();
    auto sky_opacity =
        torch::logit(1.0f * torch::ones({num_sky_points}, device));

    // Concatenate with existing gaussians
    anchors_ = torch::cat({anchors_, sky_anchor}, 0);
    scaling_ = torch::cat({scaling_, sky_scale}, 0);
    quaternion_ = torch::cat({quaternion_, sky_quat}, 0);
    opacity_ = torch::cat({opacity_, sky_opacity}, 0);
  }

  // # features
  num_points = anchors_.size(0);
  offsets_ = torch::zeros({num_points, 3}, device);
  sh_degree_to_use_ = 0;
  auto dim_sh = num_sh_bases(k_sh_degree);
  features_dc_ = torch::rand({num_points, 1, 3}, device);
  features_rest_ = torch::zeros({num_points, dim_sh - 1, 3}, device);

  auto is_nan = anchors_.isnan().any(-1) | scaling_.isnan().any(-1) |
                quaternion_.isnan().any(-1) | opacity_.isnan();
  int num_nan = is_nan.sum().item<int>();
  if (num_nan > 0) {
    std::cout << "Produce " << is_nan.sum().item<int>() << " nan splats\n";

    auto valid_idx = (~is_nan).nonzero().squeeze();
    anchors_ = anchors_.index_select(0, valid_idx).contiguous();
    scaling_ = scaling_.index_select(0, valid_idx).contiguous();
    quaternion_ = quaternion_.index_select(0, valid_idx).contiguous();
    opacity_ = opacity_.index_select(0, valid_idx).contiguous();
    features_dc_ = features_dc_.index_select(0, valid_idx).contiguous();
    features_rest_ = features_rest_.index_select(0, valid_idx).contiguous();
  }

  anchors_ = register_parameter("anchors", anchors_, false);
  offsets_ = register_parameter("offsets", offsets_);
  scaling_ = register_parameter("scaling", scaling_);
  quaternion_ = register_parameter("quaternion", quaternion_);
  opacity_ = register_parameter("opacity", opacity_);
  features_dc_ = register_parameter("features_dc", features_dc_);
  features_rest_ = register_parameter("features_rest", features_rest_);

  optimizer_params_groups_.push_back(torch::optim::OptimizerParamGroup(
      {offsets_},
      std::make_unique<torch::optim::AdamOptions>(1.6e-4f * spatial_scale_)));

  optimizer_params_groups_.push_back(torch::optim::OptimizerParamGroup(
      {scaling_}, std::make_unique<torch::optim::AdamOptions>(0.005f)));

  optimizer_params_groups_.push_back(torch::optim::OptimizerParamGroup(
      {quaternion_}, std::make_unique<torch::optim::AdamOptions>(0.001f)));
  optimizer_params_groups_.push_back(torch::optim::OptimizerParamGroup(
      {opacity_}, std::make_unique<torch::optim::AdamOptions>(0.05f)));
  optimizer_params_groups_.push_back(torch::optim::OptimizerParamGroup(
      {features_dc_}, std::make_unique<torch::optim::AdamOptions>(0.0025f)));
  optimizer_params_groups_.push_back(torch::optim::OptimizerParamGroup(
      {features_rest_},
      std::make_unique<torch::optim::AdamOptions>(0.0025f / 20.0f)));

  for (auto &group : optimizer_params_groups_) {
    dynamic_cast<torch::optim::AdamOptions &>(group.options()).eps(1e-15f);
  }
}

NeuralGS::NeuralGS(const LocalMap::Ptr &_local_map_ptr,
                   const std::filesystem::path &input_path)
    : local_map_ptr_(_local_map_ptr), torch::nn::Module() {
  load_ply_to_gs(input_path);
  sh_degree_to_use_ = k_sh_degree;
}

torch::Tensor NeuralGS::get_xyz() {
  return (anchors_ + offsets_).view({-1, 3});
}

torch::Tensor NeuralGS::get_scale() {
  return torch::exp(scaling_).view({-1, 3});
}

torch::Tensor NeuralGS::get_opacity(const bool &training) {
  if (!training && k_render_mode) {
    // https://github.com/hbb1/2d-gaussian-splatting/issues/115
    return torch::ones_like(opacity_);
  } else {
    return torch::sigmoid(opacity_);
  }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
NeuralGS::generate_gaussian(const bool &training, const torch::Tensor &view_pos,
                            const torch::Tensor &mask) {
  auto xyzs = get_xyz();
  auto quat = quaternion_.view({-1, 4});
  auto scales = get_scale();
  auto opacity = get_opacity(training);

  auto sh_feature = torch::cat({features_dc_, features_rest_}, 1);
  return std::make_tuple(xyzs, quat, scales, opacity, sh_feature,
                         torch::Tensor());
}

// _pose: [3,4]
std::map<std::string, torch::Tensor>
NeuralGS::render(const torch::Tensor &_pose_cam2world, sensor::Cameras camera,
                 const bool &training, const int &bck_color) {
  std::lock_guard<std::mutex> guard(render_mutex_);

  std::map<std::string, torch::Tensor> render_results;

  static auto device = anchors_.device();
  torch::Tensor intr_ = torch::tensor({{camera.fx, 0.0f, camera.cx},
                                       {0.0f, camera.fy, camera.cy},
                                       {0.0f, 0.0f, 1.0f}},
                                      device)
                            .unsqueeze(0); // [1,3,3]

  auto rot = _pose_cam2world.slice(1, 0, 3).to(device);
  auto pos = _pose_cam2world.slice(1, 3, 4).to(device);
  auto pose_world2cam =
      torch::cat({rot.t(), -rot.t().matmul(pos)}, 1).to(device);
  pose_world2cam =
      torch::cat(
          {pose_world2cam, torch::tensor({{0.0f, 0.0f, 0.0f, 1.0f}}, device)},
          0)
          .unsqueeze(0);

  torch::Tensor visible_mask;
  auto [xyzs, quat, scales, opacity, color_feat, selection_mask] =
      generate_gaussian(training, pos, visible_mask);
  render_results["selection_mask"] = selection_mask;

  torch::Tensor renders, alphas;
  std::map<std::string, torch::Tensor> info;
  std::tie(renders, alphas, info) = rasterization_2dgs_sdf(
      xyzs, quat, scales, opacity, color_feat, pose_world2cam, intr_,
      camera.width, camera.height, "RGB+ED", k_near, k_far, 0.0f,
      sh_degree_to_use_, true, 16, at::nullopt, false, k_use_absgrad, false,
      local_map_ptr_, training);

  torch::Tensor color, depth, sdf_normal;
  if (renders.size(-1) == 4) {
    color = renders.slice(-1, 0, 3)[0];
    depth = renders.slice(-1, 3, 4)[0];
  } else if (renders.size(-1) > 4) {
    color = renders.slice(-1, 0, 3)[0];
    depth = renders.slice(-1, 3, 4)[0];
    sdf_normal = renders.slice(-1, 4, 7)[0];
  } else {
    color = renders[0];
    depth = torch::zeros({camera.height, camera.width, 1}, device);
  }

  if (bck_color == 2) {
    auto background = torch::rand({camera.height, camera.width, 3}, device);
    render_results["color"] = color + (1.0f - alphas[0]) * background;
  } else if (bck_color == 1) {
    // white background
    render_results["color"] = color + (1.0f - alphas[0]);
  } else if (bck_color == 0) {
    // black background
    render_results["color"] = color;
  }
  render_results["depth"] = depth;
  render_results["alpha"] = alphas;
  render_results["sdf_normal"] = sdf_normal;

  render_results["xyz"] = xyzs;
  render_results.insert(info.begin(), info.end());

  if (render_results[key_for_gradient].requires_grad()) {
    render_results[key_for_gradient].retain_grad();
  }
  return render_results;
}

void NeuralGS::train_callback(
    const int &_iter, const int &_total_iter,
    const std::shared_ptr<torch::optim::Adam> &_p_optimizer,
    std::map<std::string, torch::Tensor> &info) {
  std::lock_guard<std::mutex> guard(render_mutex_);
  torch::NoGradGuard no_grad;
  auto refine_stop_iter = _total_iter / 2;
  if (!info.empty()) {
    llog::RecordValue("gs_num", anchors_.size(0), false);
    update_state(info);
    prune_nan_gs(_iter, _p_optimizer);
    prune_invisible_gs(_iter, _p_optimizer);

    // # Every 1000 its we increase the levels of SH up to a maximum degree
    sh_degree_to_use_ = min(k_sh_degree, _iter / k_sh_degree_interval);

    if (_iter < refine_stop_iter) {
      if (_iter > 0) {
        if (_iter > k_refine_start_iter) {
          if ((_iter % k_refine_every == 0) &&
              ((_iter % k_reset_every) >= pause_refine_after_reset)) {
            std::cout << "\nRefine GS at iter: " << _iter << ": ";
            grow_gs(_iter, _p_optimizer);

            // prune
            prune_gs(_iter, _p_optimizer);
            zero_state();
            std::cout << "\n";
          }
        }
        if (_iter % k_reset_every == 0) {
          std::cout << "\nReset opacity at iter: " << _iter << "\n";
          reset_opacity(_p_optimizer);
        }
      }
    }
  }

  // learning rate decay
  float iter_ratio = (float)_iter / _total_iter;

  static float gs_xyz_lr_init = 1.6e-4f * spatial_scale_;
  static float gs_xyz_lr_final = 1.6e-6f * spatial_scale_;
  // ExponentialDecayScheduler
  float gs_xyz_lr = std::exp(std::log(gs_xyz_lr_init) * (1 - iter_ratio) +
                             std::log(gs_xyz_lr_final) * iter_ratio);
  // offsets_
  _p_optimizer->param_groups()[gs_param_start_idx].options().set_lr(gs_xyz_lr);
  auto sdf_lr = k_detach_sdf_grad ? 0.0f : min(gs_xyz_lr, k_lr_end);
  for (int i = 0; i < gs_param_start_idx; i++) {
    _p_optimizer->param_groups()[i].options().set_lr(sdf_lr);
  }
}

void NeuralGS::update_state(std::map<std::string, torch::Tensor> &info) {
  // update_state
  torch::Tensor grads;
  if (k_use_absgrad) {
    grads = info["absgrad"].grad().clone();
  } else {
    grads = info[key_for_gradient].grad().clone();
  }
  int n_cameras = info["n_cameras"].item<int>();
  int width = info["width"].item<int>();
  int height = info["height"].item<int>();

  auto device = grads.device();
  int n_gaussian;
  n_gaussian = anchors_.size(0);

  if ((k_refine_scale2d_stop_iter > 0) &&
      (state.find("radii") == state.end())) {
    state["radii"] = torch::zeros({n_gaussian}, device);
  }

  if (state.find("grad2d") == state.end()) {
    state["grad2d"] = torch::zeros({n_gaussian}, device);
  }
  if (state.find("count") == state.end()) {
    state["count"] = torch::zeros({n_gaussian}, device);
  }
  if (state.find("vis") == state.end()) {
    state["vis"] = torch::zeros({n_gaussian}, device);
  }

  auto gs_ids = info["gaussian_ids"];
  static auto image_size = (float)(max(width, height));

  grads.index_put_({torch::indexing::Slice(), 0},
                   grads.select(1, 0) * width * 0.5f * n_cameras);
  grads.index_put_({torch::indexing::Slice(), 1},
                   grads.select(1, 1) * height * 0.5f * n_cameras);

  state["grad2d"].index_add_(0, gs_ids, grads.norm(2, -1));

  // put max visibilities into vis
  auto gs_vis = info["visibilities"].squeeze();
  state["vis"].index_put_(
      {gs_ids}, torch::maximum(state["vis"].index_select(0, gs_ids), gs_vis));

  auto vis_gs_idx =
      gs_ids.index_select(0, (gs_vis > 1e-4f).nonzero().squeeze());
  // std::cout << "Visible GS: " << vis_gs_idx.size(0) << "/" << gs_ids.size(0)
  //           << "; ";
  state["count"].index_add_(0, vis_gs_idx,
                            torch::ones_like(vis_gs_idx, torch::kFloat32));

  if (k_refine_scale2d_stop_iter > 0) {
    state["radii"].index_put_(
        {gs_ids}, torch::maximum(state["radii"].index_select(0, gs_ids),
                                 info["radii"] / image_size));
  }
}

void NeuralGS::zero_state() {
  state["grad2d"].zero_();
  state["count"].zero_();
  if (k_refine_scale2d_stop_iter > 0) {
    state["radii"].zero_();
  }
}

void NeuralGS::grow_gs(
    const int &iter, const std::shared_ptr<torch::optim::Adam> &_p_optimizer) {
  // grow GSs
  auto count = state["count"];
  auto grads = state["grad2d"] / count.clamp_min(1);

  auto is_grad_high = grads > k_grow_grad2d;

  auto scale = get_scale();
  scale = scale.slice(-1, 0, 2);
  auto is_small = get<0>(scale.max(-1)) <= k_grow_scale3d;
  auto is_dupli = is_grad_high & is_small;

  auto is_large = ~is_small;
  auto is_split = is_grad_high & is_large;
  if (iter < k_refine_scale2d_stop_iter) {
    is_split |= (state["radii"] > k_grow_scale2d);
  }

  // # first duplicate
  auto n_dupli = duplicate(_p_optimizer, is_dupli);

  is_split = torch::cat({
      is_split,
      torch::zeros({n_dupli}, is_split.options()),
  });

  // # then split
  split(_p_optimizer, is_split);
}

int NeuralGS::duplicate(const std::shared_ptr<torch::optim::Adam> &_p_optimizer,
                        const torch::Tensor &is_dupli) {
  auto n_dupli = is_dupli.sum().item<int>();
  if (n_dupli > 0) {
    cout << "Duplicate " << n_dupli << "; ";

    auto dupli_idx = is_dupli.nonzero().squeeze();

    auto dupli_anchors = anchors_.index_select(0, dupli_idx);
    auto dupli_offsets = offsets_.index_select(0, dupli_idx);
    auto dupli_scales = scaling_.index_select(0, dupli_idx);

    anchors_ = torch::cat({anchors_, dupli_anchors}, 0);
    cat_tensors_to_optimizer(_p_optimizer.get(), dupli_offsets, offsets_,
                             gs_param_start_idx);

    auto dupli_opacity = opacity_.index_select(0, dupli_idx);
    auto dupli_quats = quaternion_.index_select(0, dupli_idx);

    cat_tensors_to_optimizer(_p_optimizer.get(), dupli_scales, scaling_,
                             gs_param_start_idx + 1);
    cat_tensors_to_optimizer(_p_optimizer.get(), dupli_quats, quaternion_,
                             gs_param_start_idx + 2);
    cat_tensors_to_optimizer(_p_optimizer.get(), dupli_opacity, opacity_,
                             gs_param_start_idx + 3);
    cat_tensors_to_optimizer(_p_optimizer.get(),
                             features_dc_.index_select(0, dupli_idx),
                             features_dc_, gs_param_start_idx + 4);
    cat_tensors_to_optimizer(_p_optimizer.get(),
                             features_rest_.index_select(0, dupli_idx),
                             features_rest_, gs_param_start_idx + 5);

    for (auto &state_item : state) {
      state_item.second = torch::cat(
          {state_item.second, state_item.second.index_select(0, dupli_idx)});
    }
  }
  return n_dupli;
}

int NeuralGS::split(const std::shared_ptr<torch::optim::Adam> &_p_optimizer,
                    const torch::Tensor &is_split) {
  auto n_split = is_split.sum().item<int>();
  if (n_split > 0) {
    cout << "Split " << n_split << "; ";

    auto sel_idx = is_split.nonzero().squeeze();
    auto rest_idx = (~is_split).nonzero().squeeze();

    // [N, 3]
    auto scales = get_scale().index_select(0, sel_idx);
    scales = torch::cat(
        {scales.slice(-1, 0, 2), torch::zeros({n_split, 1}, scales.options())},
        1);

    static int K = 2;
    // [1, N, 3]
    auto sample_scales = scales.unsqueeze(0);
    // [K, N, 3]
    sample_scales =
        sample_scales * torch::randn({K, scales.size(0), 3}, scales.options());

    auto quats = torch::nn::functional::normalize(
        quaternion_.index_select(0, sel_idx),
        torch::nn::functional::NormalizeFuncOptions().dim(-1));
    // [N, 3, 3]
    auto rotmats = utils::normalized_quat_to_rotmat(quats);
    // [KN, 3]
    auto split_offsets =
        (torch::einsum("nij,nj,bnj->bni", {rotmats, scales, sample_scales}) +
         offsets_.index_select(0, sel_idx).unsqueeze(0))
            .reshape({-1, 3});

    auto split_anchors = anchors_.index_select(0, sel_idx).repeat({K, 1});
    auto split_opacity = opacity_.index_select(0, sel_idx).repeat({K});
    auto spilt_scales = torch::log(scales / 1.6f).repeat({K, 1});
    torch::Tensor grow_quats =
        quaternion_.index_select(0, sel_idx).repeat({K, 1});

    anchors_ =
        torch::cat({anchors_.index_select(0, rest_idx), split_anchors}, 0);
    prune_cat_tensors_to_optimizer(_p_optimizer.get(), offsets_, rest_idx,
                                   split_offsets, gs_param_start_idx);
    prune_cat_tensors_to_optimizer(_p_optimizer.get(), scaling_, rest_idx,
                                   spilt_scales, gs_param_start_idx + 1);

    prune_cat_tensors_to_optimizer(_p_optimizer.get(), quaternion_, rest_idx,
                                   grow_quats, gs_param_start_idx + 2);
    prune_cat_tensors_to_optimizer(_p_optimizer.get(), opacity_, rest_idx,
                                   split_opacity, gs_param_start_idx + 3);
    prune_cat_tensors_to_optimizer(
        _p_optimizer.get(), features_dc_, rest_idx,
        features_dc_.index_select(0, sel_idx).repeat({K, 1, 1}),
        gs_param_start_idx + 4);
    prune_cat_tensors_to_optimizer(
        _p_optimizer.get(), features_rest_, rest_idx,
        features_rest_.index_select(0, sel_idx).repeat({K, 1, 1}),
        gs_param_start_idx + 5);

    for (auto &state_item : state) {
      state_item.second =
          torch::cat({state_item.second.index_select(0, rest_idx),
                      state_item.second.index_select(0, sel_idx).repeat({K})});
    }
  }
  return n_split;
}

int NeuralGS::prune_gs(const std::shared_ptr<torch::optim::Adam> &_p_optimizer,
                       const torch::Tensor &is_prune) {
  int n_prune = is_prune.sum().item<int>();
  if (n_prune > 0) {
    auto valid_idx = (~is_prune).nonzero().squeeze();
    auto valid_num = valid_idx.numel();
    anchors_ = anchors_.index_select(0, valid_idx);
    prune_optimizer(_p_optimizer.get(), valid_idx, offsets_,
                    gs_param_start_idx);
    prune_optimizer(_p_optimizer.get(), valid_idx, scaling_,
                    gs_param_start_idx + 1);
    prune_optimizer(_p_optimizer.get(), valid_idx, quaternion_,
                    gs_param_start_idx + 2);
    prune_optimizer(_p_optimizer.get(), valid_idx, opacity_,
                    gs_param_start_idx + 3);
    prune_optimizer(_p_optimizer.get(), valid_idx, features_dc_,
                    gs_param_start_idx + 4);
    prune_optimizer(_p_optimizer.get(), valid_idx, features_rest_,
                    gs_param_start_idx + 5);

    for (auto &state_item : state) {
      state_item.second = state_item.second.index_select(0, valid_idx);
    }
  }
  return n_prune;
}

void NeuralGS::prune_gs(const int &iter,
                        const std::shared_ptr<torch::optim::Adam> &_p_optimizer,
                        const bool &prune_opa_only) {
  // only consider trained gaussians
  auto opacity = get_opacity();
  auto is_prune = opacity < k_prune_opa;
  auto n_prune_opa = is_prune.sum().item<int>();
  if (n_prune_opa > 0) {
    cout << "Prune " << n_prune_opa << " low opa; ";
  }
  auto scale = get_scale();
  scale = scale.slice(-1, 0, 2);
  // remove nan values
  auto is_too_small = get<0>(scale.min(-1)) < 1e-4f;
  auto n_prune_small = is_too_small.sum().item<int>();
  if (n_prune_small > 0) {
    is_prune |= is_too_small;
    cout << "Prune " << n_prune_small << " small; ";
  }

  if (!prune_opa_only) {
    // Dont use it, it deletes too much points in large-scale scenes
    if (iter > k_reset_every) {
      // it avoid too much small weight splat
      auto is_too_big =
          get<0>(scale.max(-1)) > k_prune_scale3d * original_spatial_scale_;
      int n_prune_big = is_too_big.sum().item<int>();
      if (n_prune_big > 0) {
        is_prune |= is_too_big;
        std::cout << "Prune " << n_prune_big << " too big splats; ";
      }
    }
  }
  int n_prune = prune_gs(_p_optimizer, is_prune);
}

void NeuralGS::prune_invisible_gs(
    const int &iter, const std::shared_ptr<torch::optim::Adam> &_p_optimizer,
    const bool &prune_opa_only) {
  if ((iter > 0) && ((iter % num_train_data_) == 0)) {
    // remove invisible splats
    auto is_prune = state["vis"] < 1e-4;
    state["vis"].zero_();

    int n_prune = prune_gs(_p_optimizer, is_prune);
    if (n_prune > 0) {
      cout << "\nPrune " << n_prune << " invisible; ";
    }
  }
}

void NeuralGS::prune_nan_gs(
    const int &iter, const std::shared_ptr<torch::optim::Adam> &_p_optimizer) {
  auto is_prune = offsets_.isnan().any(-1) | scaling_.isnan().any(-1) |
                  quaternion_.isnan().any(-1);

  int n_prune = prune_gs(_p_optimizer, is_prune);
  if (n_prune > 0) {
    cout << "Prune " << n_prune << " nan; ";
  }
}

void NeuralGS::reset_opacity(
    const std::shared_ptr<torch::optim::Adam> &_p_optimizer) {
  // opacity activation
  auto reset_opacities = opacity_.clamp_max(
      torch::logit(torch::tensor(k_prune_opa * 2.0f, opacity_.device())));

  replace_tensors_to_optimizer(_p_optimizer.get(), opacity_, reset_opacities,
                               gs_param_start_idx + 3);
}

void NeuralGS::export_gs_to_ply(std::filesystem::path &output_path,
                                const bool &export_as_3dgs) {
  std::lock_guard<std::mutex> guard(render_mutex_);

  // Make sure input tensor is on CPU
  std::string filename = output_path;

  tinyply::PlyFile export_ply;

  auto xyzs = get_xyz().detach().cpu().contiguous();
  if (xyzs.numel() > 0) {
    auto xyz = xyzs.to(torch::kFloat32);
    export_ply.add_properties_to_element(
        "vertex", {"x", "y", "z"},
        ply_utils::torch_type_to_ply_type(xyz.scalar_type()), xyz.size(0),
        reinterpret_cast<uint8_t *>(xyz.data_ptr()), tinyply::Type::INVALID, 0);
    std::cout << "Exporting properties :" << "xyz\n";
  }

  torch::Tensor f_dc;
  if (features_dc_.numel() > 0) {
    f_dc = features_dc_.detach().transpose(1, 2).flatten(1).cpu().contiguous();
    auto dim_dc = f_dc.size(-1);
    std::vector<std::string> propertyKeys;
    propertyKeys.reserve(dim_dc);
    for (int i = 0; i < dim_dc; i++) {
      propertyKeys.push_back("f_dc_" + std::to_string(i));
      std::cout << "Exporting properties :"
                << "f_dc_" + std::to_string(i) << "\n";
    }
    export_ply.add_properties_to_element(
        "vertex", propertyKeys,
        ply_utils::torch_type_to_ply_type(f_dc.scalar_type()), f_dc.size(0),
        reinterpret_cast<uint8_t *>(f_dc.data_ptr()), tinyply::Type::INVALID,
        0);
  }

  torch::Tensor f_rest;
  if (k_sh_degree > 0) {
    if (features_rest_.numel() > 0) {
      // cant move f_rest into if
      f_rest =
          features_rest_.detach().transpose(1, 2).flatten(1).cpu().contiguous();
      auto dim_sh = num_sh_bases(k_sh_degree);
      std::vector<std::string> propertyKeys;
      propertyKeys.reserve((dim_sh - 1) * 3);
      for (int i = 0; i < (dim_sh - 1) * 3; i++) {
        propertyKeys.push_back("f_rest_" + std::to_string(i));
        std::cout << "Exporting properties :"
                  << "f_rest_" + std::to_string(i) << "\n";
      }
      export_ply.add_properties_to_element(
          "vertex", propertyKeys,
          ply_utils::torch_type_to_ply_type(f_rest.scalar_type()),
          f_rest.size(0), reinterpret_cast<uint8_t *>(f_rest.data_ptr()),
          tinyply::Type::INVALID, 0);
    }
  }

  auto opacities = opacity_.detach().cpu().contiguous();
  if (opacities.numel() > 0) {
    export_ply.add_properties_to_element(
        "vertex", {"opacity"},
        ply_utils::torch_type_to_ply_type(opacities.scalar_type()),
        opacities.size(0), reinterpret_cast<uint8_t *>(opacities.data_ptr()),
        tinyply::Type::INVALID, 0);
    std::cout << "Exporting properties :"
              << "opacity\n";
  }

  torch::Tensor scale;
  if (scaling_.numel() > 0) {
    scale = scaling_.detach().cpu().contiguous();
    if (export_as_3dgs) {
      scale = torch::cat({scale.slice(-1, 0, 2),
                          torch::full({scale.size(0), 1}, log(1e-6f),
                                      scale.options())},
                         -1)
                  .contiguous();
    }
    export_ply.add_properties_to_element(
        "vertex", {"scale_0", "scale_1", "scale_2"},
        ply_utils::torch_type_to_ply_type(scale.scalar_type()), scale.size(0),
        reinterpret_cast<uint8_t *>(scale.data_ptr()), tinyply::Type::INVALID,
        0);
    std::cout << "Exporting properties :"
              << "scale\n";
  }

  torch::Tensor quat;
  if (quaternion_.numel() > 0) {
    quat = quaternion_.detach().cpu().contiguous();
    export_ply.add_properties_to_element(
        "vertex", {"rot_0", "rot_1", "rot_2", "rot_3"},
        ply_utils::torch_type_to_ply_type(quat.scalar_type()), quat.size(0),
        reinterpret_cast<uint8_t *>(quat.data_ptr()), tinyply::Type::INVALID,
        0);
    std::cout << "Exporting properties :"
              << "rot\n";
  }

  std::cout << "\033[1m\033[34m\nSaving ply to: " << output_path << "\n\033[0m";
  std::filebuf fb_ascii;
  fb_ascii.open(filename, std::ios::out);
  std::ostream outstream_ascii(&fb_ascii);
  // create directory

  if (outstream_ascii.fail())
    throw std::runtime_error("failed to open " + filename);
  // Write an ASCII file, and must be binary!
  export_ply.write(outstream_ascii, true);
  std::cout << "\033[1m\033[32m" << filename << " saved!\n\033[0m";
}

void NeuralGS::load_ply_to_gs(const std::filesystem::path &input_path) {
  std::lock_guard<std::mutex> guard(render_mutex_);
  std::cout << "Loading ply from: " << input_path << std::endl;
  std::unique_ptr<std::istream> file_stream =
      std::make_unique<std::ifstream>(input_path, std::ios::binary);

  if (!file_stream || file_stream->fail()) {
    std::cout << "file_stream failed to open " + input_path.string() << "\n";
    abort();
  }

  file_stream->seekg(0, std::ios::end);
  const float size_mb = file_stream->tellg() * float(1e-6);
  file_stream->seekg(0, std::ios::beg);

  tinyply::PlyFile file;
  file.parse_header(*file_stream);

  // Because most people have their own mesh types, tinyply treats parsed data
  // as structured/typed byte buffers. See examples below on how to marry your
  // own application-specific data structures with this one.
  std::shared_ptr<tinyply::PlyData> xyzs, f_dc, f_rest, opacities, scale, quat;

  // The header information can be used to programmatically extract properties
  // on elements known to exist in the header prior to reading the data. For
  // brevity of this sample, properties like vertex position are hard-coded:
  try {
    xyzs = file.request_properties_from_element("vertex", {"x", "y", "z"});
  } catch (const std::exception &e) {
    std::cerr << "tinyply exception: " << e.what() << "\n";
  }

  int dim_dc;
  try {
    dim_dc = 3;

    std::vector<std::string> propertyKeys;
    propertyKeys.reserve(dim_dc);
    for (int i = 0; i < dim_dc; i++) {
      propertyKeys.push_back("f_dc_" + std::to_string(i));

      std::cout << "Loading properties :" << "f_dc_" + std::to_string(i)
                << "\n";
    }
    f_dc = file.request_properties_from_element("vertex", propertyKeys);
  } catch (const std::exception &e) {
    std::cerr << "tinyply exception: " << e.what() << "\n";
  }

  if (k_sh_degree > 0) {
    try {
      auto dim_sh = num_sh_bases(k_sh_degree);
      std::vector<std::string> propertyKeys;
      propertyKeys.reserve((dim_sh - 1) * 3);
      for (int i = 0; i < (dim_sh - 1) * 3; i++) {
        propertyKeys.push_back("f_rest_" + std::to_string(i));
        std::cout << "Loading properties :" << "f_rest_" + std::to_string(i)
                  << "\n";
      }
      f_rest = file.request_properties_from_element("vertex", propertyKeys);
    } catch (const std::exception &e) {
      std::cerr << "tinyply exception: " << e.what() << "\n";
    }
  }

  try {
    opacities = file.request_properties_from_element("vertex", {"opacity"});
  } catch (const std::exception &e) {
    std::cerr << "tinyply exception: " << e.what() << "\n";
  }

  try {
    scale = file.request_properties_from_element(
        "vertex", {"scale_0", "scale_1", "scale_2"});
  } catch (const std::exception &e) {
    std::cerr << "tinyply exception: " << e.what() << "\n";
  }

  try {
    quat = file.request_properties_from_element(
        "vertex", {"rot_0", "rot_1", "rot_2", "rot_3"});
  } catch (const std::exception &e) {
    std::cerr << "tinyply exception: " << e.what() << "\n";
  }

  file.read(*file_stream);

  if (xyzs->t == tinyply::Type::FLOAT32) {
    anchors_ = torch::from_blob(xyzs->buffer.get(), {(long)xyzs->count, 3},
                                torch::kFloat32)
                   .clone()
                   .to(k_device)
                   .contiguous();
    offsets_ = torch::zeros_like(anchors_);
  }

  if (f_dc->t == tinyply::Type::FLOAT32) {
    features_dc_ =
        torch::from_blob(f_dc->buffer.get(), {(long)f_dc->count, dim_dc, 1},
                         torch::kFloat32)
            .clone()
            .to(k_device)
            .transpose(1, 2)
            .contiguous();
  }

  if (k_sh_degree > 0) {
    if (f_rest->t == tinyply::Type::FLOAT32) {
      auto dim_sh = num_sh_bases(k_sh_degree);
      features_rest_ = torch::from_blob(f_rest->buffer.get(),
                                        {(long)f_rest->count, 3, dim_sh - 1},
                                        torch::kFloat32)
                           .clone()
                           .to(k_device)
                           .transpose(1, 2)
                           .contiguous();
    }
  } else {
    features_rest_ = torch::zeros({anchors_.size(0), 0, 3}, k_device);
  }

  if (opacities->t == tinyply::Type::FLOAT32) {
    opacity_ = torch::from_blob(opacities->buffer.get(),
                                {(long)opacities->count}, torch::kFloat32)
                   .clone()
                   .to(k_device)
                   .contiguous();
  }

  if (scale->t == tinyply::Type::FLOAT32) {
    scaling_ = torch::from_blob(scale->buffer.get(), {(long)scale->count, 3},
                                torch::kFloat32)
                   .clone()
                   .to(k_device)
                   .contiguous();
  }

  if (quat->t == tinyply::Type::FLOAT32) {
    quaternion_ = torch::from_blob(quat->buffer.get(), {(long)quat->count, 4},
                                   torch::kFloat32)
                      .clone()
                      .to(k_device)
                      .contiguous();
  }

  std::cout << "Loaded " << input_path << " with " << anchors_.size(0)
            << " gaussians\n";
}

void NeuralGS::freeze_structure() {
  std::lock_guard<std::mutex> guard(render_mutex_);
  offsets_.set_requires_grad(false);
  scaling_.set_requires_grad(false);
  quaternion_.set_requires_grad(false);
  opacity_.set_requires_grad(false);
}

void NeuralGS::unfreeze_structure() {
  std::lock_guard<std::mutex> guard(render_mutex_);
  offsets_.set_requires_grad(true);
  scaling_.set_requires_grad(true);
  quaternion_.set_requires_grad(true);
  opacity_.set_requires_grad(true);
}