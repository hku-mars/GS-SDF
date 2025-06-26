#include "neural_mapping.h"

#ifdef ENABLE_ROS
#include <tf/transform_broadcaster.h>
#endif

#include <opencv2/opencv.hpp>

#include "llog/llog.h"
#include "optimizer/loss.h"
#include "optimizer/loss_utils/loss_utils.h"
#include "params/params.h"
#include "utils/sensor_utils/cameras.hpp"
#include "utils/tqdm.hpp"

#include "kaolin_wisp_cpp/spc_ops/spc_ops.h"
#include "utils/coordinates.h"

#include "utils/ply_utils/ply_utils_torch.h"

using namespace std;

NeuralSLAM::NeuralSLAM(const int &mode,
                       const std::filesystem::path &_config_path,
                       const std::filesystem::path &_data_path) {
  cout << "TORCH_VERSION: " << TORCH_VERSION << '\n';
  cout << "Config Path: " << _config_path << '\n';

  read_params(_config_path, _data_path, mode);

  data_loader_ptr = std::make_unique<dataloader::DataLoader>(
      k_dataset_path, k_dataset_type, k_device, mode & k_preload, k_res_scale,
      k_sensor, k_ds_pt_num, k_max_time_diff_camera_and_pose,
      k_max_time_diff_lidar_and_pose);

  if (k_export_colmap_format == 1) {
    data_loader_ptr->export_as_colmap_format(false, false);
  } else if (k_export_colmap_format == 2) {
    data_loader_ptr->export_as_colmap_format_for_nerfstudio(false);
  } else if (k_export_colmap_format == 3) {
    data_loader_ptr->export_as_colmap_format(true, false);
  }

  if (mode) {
    mapper_thread = std::thread(&NeuralSLAM::batch_train, this);
    keyboard_thread = std::thread(&NeuralSLAM::keyboard_loop, this);
    misc_thread = std::thread(&NeuralSLAM::misc_loop, this);
  } else {
    // old_k_output_path/model
    k_model_path = _config_path.parent_path().parent_path().parent_path();
    k_output_path = k_model_path.parent_path();
    load_pretrained(k_model_path);
  }
}

void NeuralSLAM::load_pretrained(
    const std::filesystem::path &_pretrained_path) {
  load_checkpoint(_pretrained_path);

  if (neural_gs_ptr) {
    neural_gs_ptr->sh_degree_to_use_ = k_sh_degree;
  }

#ifdef ENABLE_ROS
  mapper_thread = std::thread(&NeuralSLAM::pretrain_loop, this);
#endif
  keyboard_thread = std::thread(&NeuralSLAM::keyboard_loop, this);
}

DepthSamples NeuralSLAM::sample(DepthSamples _ray_samples,
                                const float &sample_std,
                                const bool &_sample_free) {
  static auto p_t_utils_sample = llog::CreateTimer("   utils::sample");
  p_t_utils_sample->tic();
  _ray_samples.ridx = torch::arange(_ray_samples.size(0), k_device);
  _ray_samples.ray_sdf = torch::zeros({_ray_samples.size(0), 1}, k_device);
  // [n,num_samples]
  DepthSamples point_samples =
      local_map_ptr->sample(_ray_samples, 1, _sample_free);

  auto surface_samples =
      utils::sample_surface_pts(_ray_samples, k_surface_sample_num, sample_std);
  point_samples = point_samples.cat(surface_samples);

  auto trunc_idx = (point_samples.ray_sdf.abs() > k_truncated_dis)
                       .squeeze()
                       .nonzero()
                       .squeeze();
  point_samples.ray_sdf.index_put_(
      {trunc_idx},
      point_samples.ray_sdf.index({trunc_idx}).sign() * k_truncated_dis);

  point_samples = point_samples.cat(_ray_samples);

  // if contraction no need to filter
  auto inrange_idx =
      local_map_ptr->get_inrange_mask(point_samples.xyz).nonzero().squeeze();
  point_samples = point_samples.index_select(0, inrange_idx);
  p_t_utils_sample->toc_sum();
  return point_samples;
}

torch::Tensor NeuralSLAM::sdf_regularization(const torch::Tensor &xyz,
                                             const torch::Tensor &pred_sdf,
                                             const bool &curvate_enable,
                                             const std::string &name) {
  auto grad_results = local_map_ptr->get_gradient(
      xyz, k_sample_std, pred_sdf, curvate_enable, k_numerical_grad);

  auto point_grad = grad_results[0];
  auto eik_loss = loss::eikonal_loss(point_grad, name);
  auto loss = k_eikonal_weight * eik_loss;
  llog::RecordValue(name + "_eik", eik_loss.item<float>());

  if (curvate_enable) {
    auto curv_loss = loss::curvate_loss(grad_results[1], name);
    loss += k_curvate_weight * curv_loss;
    llog::RecordValue(name + "_curv", curv_loss.item<float>());
  }

  bool align_ana_num = k_align_weight > 0;
  if (!k_numerical_grad & align_ana_num) {
    auto grad_results =
        local_map_ptr->get_gradient(xyz, k_sample_std, pred_sdf, false, true);
    auto point_numerical_grad = grad_results[0].detach();

    auto align_loss = (point_grad - point_numerical_grad).abs().mean();
    loss += k_align_weight * align_loss;
    llog::RecordValue(name + "_align", align_loss.item<float>());
  }

  return loss;
}

std::tuple<torch::Tensor, DepthSamples>
NeuralSLAM::sdf_train_batch_iter(const int &iter, const bool &_sample_free) {
  static auto p_t_sample = llog::CreateTimer("  sample");
  p_t_sample->tic();

  auto batch_num =
      k_numerical_grad ? max((int)(k_batch_num / 6.f), 1) : k_batch_num;
  auto indices =
      (torch::rand({k_batch_num}) *
       data_loader_ptr->dataparser_ptr_->train_depth_pack_.depth.size(0))
          .to(torch::kLong)
          .clamp(0,
                 data_loader_ptr->dataparser_ptr_->train_depth_pack_.depth.size(
                     0) -
                     1);

  auto struct_ray_samples =
      data_loader_ptr->dataparser_ptr_->train_depth_pack_.index({indices}).to(
          k_device);

  auto point_samples = sample(struct_ray_samples, k_sample_std, _sample_free);
  auto inrange_idx = local_map_ptr->get_inrange_mask(point_samples.xyz)
                         .squeeze()
                         .nonzero()
                         .squeeze();
  point_samples = point_samples.index_select(0, inrange_idx);
  p_t_sample->toc_sum();

  static auto p_t_get_sdf = llog::CreateTimer("  get_sdf");
  p_t_get_sdf->tic();
  if (k_eikonal_weight > 0.0f && !k_numerical_grad) {
    point_samples.xyz.requires_grad_(true);
  }
  auto sdf_results = local_map_ptr->get_sdf(point_samples.xyz);
  point_samples.pred_sdf = sdf_results[0];
  point_samples.pred_isigma = sdf_results[1];

  auto sdf_loss = loss::sdf_loss(point_samples.pred_sdf, point_samples.ray_sdf,
                                 point_samples.pred_isigma);
  auto loss = k_sdf_weight * sdf_loss;
  llog::RecordValue("sdf", sdf_loss.item<float>(), true);
  p_t_get_sdf->toc_sum();

  if (k_eikonal_weight > 0.0) {
    static bool curvate_enable = k_curvate_weight > 0.0;
    loss += sdf_regularization(point_samples.xyz, point_samples.pred_sdf,
                               curvate_enable, "pt");
  }

  return std::make_tuple(loss, point_samples);
}

std::tuple<torch::Tensor, std::map<std::string, torch::Tensor>>
NeuralSLAM::gs_train_batch_iter(const int &iter, const bool &opt_struct) {
  auto gs_loss = torch::tensor(0.0f, k_device);

  auto color_loss = torch::tensor(0.0f, k_device);
  auto dssim_loss = torch::tensor(0.0f, k_device);
  auto dist_loss = torch::tensor(0.0f, k_device);
  auto normal_error = torch::tensor(0.0f, k_device);
  auto isotropic_loss = torch::tensor(0.0f, k_device);

  static auto p_t_get_rgb = llog::CreateTimer("  get_rgb");
  p_t_get_rgb->tic();

  static auto p_t_get_data = llog::CreateTimer("   get_data");
  p_t_get_data->tic();
  static auto train_num =
      data_loader_ptr->dataparser_ptr_->size(dataparser::DataType::TrainColor);
  static torch::Tensor train_cameras_idx;
  if (iter % train_num == 0) {
    train_cameras_idx = torch::randperm(train_num);
  }

  auto img_idx = train_cameras_idx[iter % train_num];

  // [3, 4]
  auto color_pose = data_loader_ptr->dataparser_ptr_
                        ->get_pose(img_idx, dataparser::DataType::TrainColor)
                        .squeeze(0)
                        .to(k_device);
  auto gt_color = data_loader_ptr->dataparser_ptr_
                      ->get_image(img_idx, dataparser::DataType::TrainColor)
                      .squeeze(0)
                      .to(k_device)
                      .contiguous();
  p_t_get_data->toc_sum();

  static auto p_t_render = llog::CreateTimer("   render");
  p_t_render->tic();
  auto render_results = neural_gs_ptr->render(
      color_pose, data_loader_ptr->dataparser_ptr_->sensor_.camera, true,
      k_bck_color);

  render_results["color_pose"] = color_pose;
  p_t_render->toc_sum();

  static auto p_t_loss = llog::CreateTimer("   loss");
  p_t_loss->tic();
  auto render_color = render_results["color"]; // dont clamp in training

  torch::Tensor mask;
  if (data_loader_ptr->dataparser_ptr_->mask.defined()) {
    mask = data_loader_ptr->dataparser_ptr_->mask.to(render_color.device());
  }

  color_loss = loss::rgb_loss(render_color, gt_color, mask);
  gs_loss += k_rgb_weight * color_loss;
  dssim_loss = loss::dssim_loss(render_color, gt_color, mask);
  gs_loss += k_dssim_weight * dssim_loss;
  p_t_loss->toc_sum();

  if (opt_struct) {
    if ((k_render_normal_weight > 0.0f) &&
        (iter > k_refine_gs_struct_start_iter)) {
      // [H, W, 1]
      torch::Tensor render_depth;
      if (k_depth_type == 0) {
        render_depth = render_results["depth"];
      } else {
        render_depth = render_results["render_median"][0];
      }
      auto depth_normal = sensor::depth_to_normal(
          data_loader_ptr->dataparser_ptr_->sensor_.camera, color_pose,
          render_depth);

      auto render_alpha = render_results["alpha"][0].detach();
      depth_normal = depth_normal * render_alpha;

      auto render_normal = render_results["render_normal"][0];
      normal_error =
          (1.0f - (depth_normal * render_normal).sum(-1).nan_to_num()).mean();
      gs_loss += k_render_normal_weight * normal_error;
    }
  }

  if (k_isotropic_weight > 0) {
    auto gaussian_ids = render_results["gaussian_ids"];
    auto scale = neural_gs_ptr->get_scale().index_select(0, gaussian_ids);

    scale = scale.slice(-1, 0, 2);

    isotropic_loss = (scale - scale.mean(-1, true)).abs().mean();
    gs_loss += k_isotropic_weight * isotropic_loss;
  }

  p_t_get_rgb->toc_sum();

  llog::RecordValue("color", color_loss.item<float>(), true);
  llog::RecordValue("dssim", dssim_loss.item<float>(), true);

  if (k_render_normal_weight > 0.0f) {
    llog::RecordValue("normal", normal_error.item<float>(), true);
  }

  if (k_isotropic_weight > 0) {
    llog::RecordValue("isotropic", isotropic_loss.item<float>(), true);
  }

  return std::make_tuple(gs_loss, render_results);
}

void NeuralSLAM::nsdf_train(int _opt_iter) {
  llog::Reset();
  log_file_ = "nsdf_log.txt";
  std::string value_path = k_output_path / log_file_;
  k_batch_num = k_batch_ray_num;
  k_sample_pts_per_ray = k_batch_pt_num / (float)k_batch_num;
  // clear VRAM to see actual memory usage
  c10::cuda::CUDACachingAllocator::emptyCache();
  auto iter_bar = tq::trange(_opt_iter);
  for (int i : iter_bar) {
    static auto p_t_iter = llog::CreateTimer("  iter");
    p_t_iter->tic();
    torch::GradMode::set_enabled(true);
    auto [loss, point_samples] = sdf_train_batch_iter(i);

    static auto p_t_backward = llog::CreateTimer("  backward");
    p_t_backward->tic();
    static torch::autograd::AnomalyMode anomaly_mode;
    anomaly_mode.set_enabled(k_debug);
    p_optimizer_->zero_grad();
    loss.backward();
    p_optimizer_->step();
    p_t_backward->toc_sum();

    p_t_iter->toc_sum();
    static auto p_t_cbk = llog::CreateTimer("  cbk");
    p_t_cbk->tic();

    torch::NoGradGuard no_grad;

    float pt_n = point_samples.size(0);
    float sample_pts_per_ray = pt_n / (float)k_batch_num;
    k_sample_pts_per_ray =
        k_sample_pts_per_ray * 0.9 + sample_pts_per_ray * 0.1;
    llog::RecordValue("s_pts_per_ray", k_sample_pts_per_ray);
    k_batch_num =
        min((int)(k_batch_pt_num / k_sample_pts_per_ray), (int)k_batch_pt_num);

    llog::RecordValue("s_ray_n", k_batch_num);
    llog::RecordValue("s_pt_n", pt_n);
    sdf_train_callback(i, k_sdf_iter_step, point_samples);
    p_t_cbk->toc_sum();

    llog::RecordValue("loss", loss.item<float>(), true);
    iter_bar << llog::FlashValue(value_path, 3);

#ifdef ENABLE_ROS
    if ((i + 1) % k_vis_frame_step == 0) {
      static auto p_t_vis = llog::CreateTimer("  vis");
      p_t_vis->tic();
      visualization(point_samples.xyz);
      p_t_vis->toc_sum();
    }
#endif

    if (i % k_export_interval == 0) {
      misc_trigger = i;
    }
  }
  // end();
}

void NeuralSLAM::gs_train(int _opt_iter) {
  llog::Reset();
  log_file_ = "gs_log.txt";
  std::string value_path = k_output_path / log_file_;
  k_batch_num = k_batch_ray_num;
  k_sample_pts_per_ray = k_batch_pt_num / (float)k_batch_num;
  // clear VRAM to see actual memory usage
  c10::cuda::CUDACachingAllocator::emptyCache();

  static torch::autograd::AnomalyMode anomaly_mode;
  anomaly_mode.set_enabled(k_debug);

  static auto train_num =
      data_loader_ptr->dataparser_ptr_->size(dataparser::DataType::TrainColor);

  if (k_color_init) {
    // init color
    neural_gs_ptr->freeze_structure();
    for (auto &param_group : p_optimizer_->param_groups()) {
      param_group.options().set_lr(10 * param_group.options().get_lr());
    }
    auto init_iter_bar = tq::trange(train_num);
    init_iter_bar.set_prefix("Color init.");
    for (int i : init_iter_bar) {
      auto [gs_loss, render_results] = gs_train_batch_iter(i, false);
      p_optimizer_->zero_grad();
      gs_loss.backward();
      p_optimizer_->step();
    }
    for (auto &param_group : p_optimizer_->param_groups()) {
      param_group.options().set_lr(0.1f * param_group.options().get_lr());
    }
    neural_gs_ptr->unfreeze_structure();
  }

  if (k_detach_sdf_grad) {
    local_map_ptr->freeze_net();
  }

  // reset sdf network lr
  std::map<std::string, torch::Tensor> empty_map;
  neural_gs_ptr->train_callback(0, k_gs_iter_step, p_optimizer_, empty_map);
  misc_trigger = 0;
  auto iter_bar = tq::trange(_opt_iter);
  for (int i : iter_bar) {
    static auto p_t_iter = llog::CreateTimer("  iter");
    p_t_iter->tic();
    torch::GradMode::set_enabled(true);

    auto sdf_nn_loss = torch::tensor(0.0f, k_device);
    DepthSamples point_samples;
    if (k_gs_sdf_reg && !k_detach_sdf_grad) {
      std::tie(sdf_nn_loss, point_samples) = sdf_train_batch_iter(i);
    }

    auto [gs_loss, render_results] = gs_train_batch_iter(i);

    auto joint_loss = torch::tensor(0.0f, k_device);

    auto gs_sdf_error = torch::tensor(0.0f, k_device);
    auto sdf_3d_normal_loss = torch::tensor(0.0f, k_device);
    auto gs_eik_loss = torch::tensor(0.0f, k_device);
    auto gs_curv_loss = torch::tensor(0.0f, k_device);
    auto gs_align_loss = torch::tensor(0.0f, k_device);
    int vis_gs_num = 0;

    RaySamples gs_raysamples;
    if (k_gs_sdf_reg && (i >= k_refine_gs_struct_start_iter)) {
      static auto p_t_opt = llog::CreateTimer("   opt");
      p_t_opt->tic();
      auto gs_samples = render_results["samples"];
      // samples' Gaussian weights
      auto gs_samples_gs_weights = render_results["samples_weights"];
      auto gs_visibilities = render_results["visibilities"].detach();

      auto gs_samples_weights =
          (gs_samples_gs_weights * gs_visibilities).detach();
      auto vis_mask = (gs_visibilities > k_visible_thr).squeeze();

      auto valid_mask = local_map_ptr->get_valid_mask(gs_samples) & vis_mask;
      if (valid_mask.sum().item<int>() > 0) {
        auto valid_gs_sample_ids = valid_mask.squeeze().nonzero().squeeze();
        gs_samples = gs_samples.index_select(0, valid_gs_sample_ids);
        gs_samples_weights =
            gs_samples_weights.index_select(0, valid_gs_sample_ids);

        vis_gs_num = gs_samples.size(0);

        static auto p_t_sdf = llog::CreateTimer("   sdf");
        p_t_sdf->tic();
        auto gs_samples_sdf_results = local_map_ptr->get_sdf(gs_samples);
        auto gs_samples_sdf = gs_samples_sdf_results[0];
        auto gs_samples_isigma = gs_samples_sdf_results[1];
        gs_raysamples.pred_isigma = gs_samples_isigma;
        p_t_sdf->toc_sum();
        p_t_opt->toc_sum();

        torch::Tensor gs_grad;
        if (k_eikonal_weight > 0.0f && !k_detach_sdf_grad) {
          sdf_nn_loss += sdf_regularization(gs_samples.detach(), gs_samples_sdf,
                                            false, "gs");
        }

        if (k_gs_sdf_weight > 0) {
          static auto p_t_gs_sdf = llog::CreateTimer("   gs_sdf");
          p_t_gs_sdf->tic();
          gs_sdf_error = loss::gs_sdf_loss(gs_samples_sdf, gs_samples_weights);
          gs_loss += k_gs_sdf_weight * gs_sdf_error;
          p_t_gs_sdf->toc_sum();
        }
      }
    }

    static auto p_t_backward = llog::CreateTimer("  backward");
    p_t_backward->tic();
    p_optimizer_->zero_grad();
    auto loss = gs_loss + sdf_nn_loss + joint_loss;
    loss.backward();
    p_optimizer_->step();
    p_t_backward->toc_sum();

    p_t_iter->toc_sum();
    static auto p_t_cbk = llog::CreateTimer("  cbk");
    p_t_cbk->tic();

    torch::NoGradGuard no_grad;

    static auto p_t_cbk_sdf = llog::CreateTimer("   cbk_sdf");
    p_t_cbk_sdf->tic();
    if (k_gs_sdf_reg && !k_detach_sdf_grad) {
      sdf_train_callback(i, k_gs_iter_step, point_samples, false);
    }
    p_t_cbk_sdf->toc_sum();
    static auto p_t_cbk_gs = llog::CreateTimer("   cbk_gs");
    p_t_cbk_gs->tic();
    neural_gs_ptr->train_callback(i, k_gs_iter_step, p_optimizer_,
                                  render_results);
    p_t_cbk_gs->toc_sum();

    if (i % k_export_interval == 0) {
      auto psnr = export_test_image(false, k_test_idx, std::to_string(i) + "_");
      llog::RecordValue("psnr", psnr);

      if (data_loader_ptr->dataparser_ptr_->eval_color_poses_.numel() > 0) {
        auto ex_psnr =
            export_test_image(true, k_test_idx, std::to_string(i) + "_ex_");
        llog::RecordValue("ex_psnr", ex_psnr);
      }
    }

    p_t_cbk->toc_sum();

    if (k_sdf_weight > 0) {
      if (k_gs_sdf_weight > 0) {
        llog::RecordValue("gs_sdf", gs_sdf_error.item<float>(), true);

        llog::RecordValue("vis_n", vis_gs_num);
      }
    }

    llog::RecordValue("gs_loss", gs_loss.item<float>(), true);
    if (!k_detach_sdf_grad) {
      llog::RecordValue("sdf_nn_loss", sdf_nn_loss.item<float>(), true);
    }
    iter_bar << llog::FlashValue(value_path, 3);

#ifdef ENABLE_ROS
    if ((i + 1) % k_vis_frame_step == 0) {
      static auto p_t_vis = llog::CreateTimer("  vis");
      p_t_vis->tic();
      visualization();
      p_t_vis->toc_sum();
    }
#endif

    if ((i % k_export_interval == 0) && (i < k_gs_iter_step)) {
      misc_trigger = i;
    }
  }
  // end();
}

void NeuralSLAM::sdf_train_callback(const int &_iter, const int &_total_iter,
                                    const RaySamples &_point_samples,
                                    const bool &_update_lr) {
  torch::NoGradGuard no_grad;
  if (_iter == -1) {
    k_sample_std = k_bce_sigma;
    return;
  }

  float iter_ratio = (float)_iter / _total_iter;

  if (_point_samples.pred_isigma.numel() > 0) {
    k_sample_std = (1.0 / _point_samples.pred_isigma).mean().item<float>();
    llog::RecordValue("sstd", k_sample_std, true);
    k_sample_std = max(k_sample_std, k_bce_sigma);
  }

  if (_update_lr) {
    float lr = k_lr * (1 - iter_ratio) + k_lr_end * iter_ratio;
    for (auto &pg : p_optimizer_->param_groups()) {
      if (pg.has_options()) {
        pg.options().set_lr(lr);
      }
    }
  }

  if (k_outlier_remove && (_iter > 0)) {
    if (_iter % k_outlier_removal_interval == 0) {
      static auto p_t_outlier_remove = llog::CreateTimer(" outlier_remove");
      p_t_outlier_remove->tic();
      // batch remove outlier
      std::vector<torch::Tensor> inlier_idx_vec;
      auto tmp_outlier_dist = exp(log(k_truncated_dis) * (1 - iter_ratio) +
                                  log(k_outlier_dist) * iter_ratio);
      auto train_pcl =
          data_loader_ptr->dataparser_ptr_->train_depth_pack_.xyz.view({-1, 3});
      int point_num = train_pcl.size(0);
      for (int i = 0; i < point_num; i += k_vis_batch_pt_num) {
        auto end = min(i + k_vis_batch_pt_num, point_num - 1);
        auto xyz_pred_sdf = local_map_ptr->get_sdf(
            train_pcl.index_select(0, {torch::arange(i, end)}).to(k_device))[0];
        auto inlier_idx = ((xyz_pred_sdf.abs() < tmp_outlier_dist).squeeze())
                              .nonzero()
                              .squeeze()
                              .cpu();
        inlier_idx_vec.emplace_back(inlier_idx + i);
      }
      auto inlier_idx = torch::cat(inlier_idx_vec, 0);
      cout << "\nOutlier Removal(" << tmp_outlier_dist << "): "
           << data_loader_ptr->dataparser_ptr_->train_depth_pack_.size(0);

      data_loader_ptr->dataparser_ptr_->train_depth_pack_ =
          data_loader_ptr->dataparser_ptr_->train_depth_pack_.index_select(
              0, inlier_idx);
      cout << " -> "
           << data_loader_ptr->dataparser_ptr_->train_depth_pack_.size(0)
           << '\n';
      p_t_outlier_remove->toc_sum();
    }
  }
}

void NeuralSLAM::prefilter_data(const bool &export_img) {
  std::vector<int> valid_ids_vec;
  std::vector<std::filesystem::path> filtered_train_color_filelists;
  int color_size =
      data_loader_ptr->dataparser_ptr_->size(dataparser::DataType::TrainColor);

  // [H, W, 3]
  auto pre_color = data_loader_ptr->dataparser_ptr_
                       ->get_image(0, dataparser::DataType::TrainColor)
                       .to(k_device);
  valid_ids_vec.emplace_back(0);
  filtered_train_color_filelists.emplace_back(
      data_loader_ptr->dataparser_ptr_->get_file(
          0, dataparser::DataType::TrainColor));

  std::filesystem::path fileter_color_path = k_output_path / "filter";
  if (export_img) {
    std::filesystem::create_directories(fileter_color_path);
    auto gt_color_file_name =
        data_loader_ptr->dataparser_ptr_
            ->get_file(0, dataparser::DataType::TrainColor)
            .filename()
            .replace_extension(".png");
    cv::imwrite(fileter_color_path / gt_color_file_name,
                utils::tensor_to_cv_mat(pre_color));
  }

  auto iter_bar = tq::trange(1, color_size);
  iter_bar.set_prefix("Prefilter Data");
  for (int i : iter_bar) {
    auto now_color = data_loader_ptr->dataparser_ptr_
                         ->get_image(i, dataparser::DataType::TrainColor)
                         .to(k_device);
    // auto metric = loss_utils::ssim(pre_color.permute({2, 0, 1}).unsqueeze(0),
    //                              now_color.permute({2, 0, 1}).unsqueeze(0))
    //                 .item<float>();

    auto metric = loss_utils::psnr(pre_color.permute({2, 0, 1}).unsqueeze(0),
                                   now_color.permute({2, 0, 1}).unsqueeze(0));
    if (metric < k_prefilter) {
      valid_ids_vec.emplace_back(i);
      filtered_train_color_filelists.emplace_back(
          data_loader_ptr->dataparser_ptr_->get_file(
              i, dataparser::DataType::TrainColor));
      pre_color = now_color;

      if (export_img) {
        auto gt_color_file_name =
            data_loader_ptr->dataparser_ptr_
                ->get_file(i, dataparser::DataType::TrainColor)
                .filename()
                .replace_extension(".png");
        cv::imwrite(fileter_color_path / gt_color_file_name,
                    utils::tensor_to_cv_mat(pre_color));
      }
    }
  }

  data_loader_ptr->dataparser_ptr_->train_color_filelists_ =
      filtered_train_color_filelists;

  auto valid_ids = torch::tensor(valid_ids_vec);
  auto reshape_color_pose_sizes =
      data_loader_ptr->dataparser_ptr_->train_color_poses_.sizes().vec();
  reshape_color_pose_sizes[0] = -1;
  data_loader_ptr->dataparser_ptr_->train_color_poses_ =
      data_loader_ptr->dataparser_ptr_->train_color_poses_
          .index_select(0, valid_ids)
          .reshape(reshape_color_pose_sizes);
  if (k_preload) {
    auto reshape_color_sizes =
        data_loader_ptr->dataparser_ptr_->train_color_.sizes().vec();
    reshape_color_sizes[0] = -1;
    data_loader_ptr->dataparser_ptr_->train_color_ =
        data_loader_ptr->dataparser_ptr_
            ->get_image(valid_ids, dataparser::DataType::TrainColor)
            .reshape(reshape_color_sizes);
  }

  std::cout << "Original color size: " << color_size << ", after filtering: "
            << data_loader_ptr->dataparser_ptr_->size(
                   dataparser::DataType::TrainColor)
            << "\n";
}

/**
 * @brief Builds an occupancy map from depth and color data
 *
 * This function processes raw depth data to create an occupancy map
 * representation. If probability mapping is enabled, it uses ROG-Map for
 * enhanced occupancy mapping. The function supports two main operations:
 * 1. calculate pcl distribution to determine the local map's center and radius
 * 2. create the occupancy map and neural field
 * 3. Depth raycasting: Creates points from depth measurements
 *
 * @return bool Success status of the map building operation
 */
bool NeuralSLAM::build_occ_map() {
  std::cout << "Starting occupancy map building from raw data...\n";
  static auto timer_cal_prior = llog::CreateTimer("cal_prior");
  timer_cal_prior->tic();

  auto pcl_depth =
      data_loader_ptr->dataparser_ptr_->train_depth_pack_.depth.view({-1});
  auto valid_mask = (pcl_depth > k_min_range) & (pcl_depth < k_max_range);
  auto valid_idx = valid_mask.nonzero().squeeze();

  auto pcl =
      data_loader_ptr->dataparser_ptr_->train_depth_pack_.xyz.view({-1, 3})
          .index_select(0, valid_idx);
  // calculate pcl's center
  // and radius
  auto pcl_center = pcl.mean(0).squeeze();
  auto pcl_radius = (pcl - pcl_center).norm(2, 1).max().item<float>();

  std::cout << "PCL center: " << pcl_center << ", radius: " << pcl_radius
            << '\n';

  // set local map's center and radius
  k_map_origin = pcl_center;
  if (k_inner_map_size < pcl_radius * 2.0f) {
    std::cout << "Warning: inner map size is smaller than pcl radius * 2.0\n";
  } else {
    k_inner_map_size = pcl_radius * 2.0f;
  }
  k_x_max = 0.5f * k_inner_map_size;
  k_y_max = k_x_max;
  k_z_max = k_x_max;
  k_x_min = -0.5f * k_inner_map_size;
  k_y_min = k_x_min;
  k_z_min = k_x_min;
  k_octree_level =
      ceil(log2((k_inner_map_size + 2 * k_leaf_size) * k_leaf_size_inv));
  k_map_resolution = std::pow(2, k_octree_level);
  k_map_size = k_map_resolution * k_leaf_size;
  k_map_size_inv = 1.0f / k_map_size;

  std::cout << "Leaf size: " << k_leaf_size
            << "Octree level: " << k_octree_level << std::endl;

  local_map_ptr = std::make_shared<LocalMap>(
      k_map_origin, k_x_min, k_x_max, k_y_min, k_y_max, k_z_min, k_z_max);
  for (auto &p : local_map_ptr->named_parameters()) {
    cout << p.key() << p.value().sizes() << '\n';
  }

  auto inrange_idx =
      local_map_ptr->get_inrange_mask(pcl).squeeze().nonzero().squeeze();
  auto inrange_pt = pcl.index_select(0, inrange_idx).cuda();

  // Update octree with combined points
  local_map_ptr->update_octree_as(inrange_pt);
  std::cout << "Updated octree with inrange points.\n";

  //----------------------------------------------------------------------
  // SECTION 1: Process depth data for occupancy mapping
  //----------------------------------------------------------------------

  // If probability map is disabled, just use depth points directly
#ifdef ENABLE_ROS
  pub_pointcloud(pointcloud_pub, inrange_pt);
#endif
  local_map_ptr->update_octree_as(inrange_pt, false, 1);

  //----------------------------------------------------------------------
  // SECTION 4: Extract and save final points from octree
  //----------------------------------------------------------------------

  // Get quantized points from acceleration structure
  auto dense_voxels = local_map_ptr->p_acc_strcut_occ_->get_quantized_points();
  // Convert quantized points to normalized floating points
  auto normalized_points =
      spc_ops::quantized_points_to_fpoints(dense_voxels, k_octree_level);
  // Convert normalized points to world coordinates
  auto world_points = local_map_ptr->m1p1_pts_to_xyz(normalized_points);
  // Export points to PLY file
  ply_utils::export_to_ply(k_model_path / "as_occ_prior.ply",
                           world_points.cpu());

  timer_cal_prior->toc_sum();

  //----------------------------------------------------------------------
  // SECTION 5: Reshape depth data for further processing
  //----------------------------------------------------------------------

  // Reshape origin data
  data_loader_ptr->dataparser_ptr_->train_depth_pack_.origin =
      data_loader_ptr->dataparser_ptr_->train_depth_pack_.origin.unsqueeze(1)
          .repeat(
              {1,
               data_loader_ptr->dataparser_ptr_->train_depth_pack_.depth.size(
                   1),
               1})
          .view({-1, 3});
  data_loader_ptr->dataparser_ptr_->train_depth_pack_.depth =
      data_loader_ptr->dataparser_ptr_->train_depth_pack_.depth.view({-1, 1});
  data_loader_ptr->dataparser_ptr_->train_depth_pack_.direction =
      data_loader_ptr->dataparser_ptr_->train_depth_pack_.direction.view(
          {-1, 3});
  data_loader_ptr->dataparser_ptr_->train_depth_pack_.xyz =
      data_loader_ptr->dataparser_ptr_->train_depth_pack_.xyz.view({-1, 3});

  if (k_export_train_pcl) {
    std::string train_points_ply_file = k_output_path / "train_points.ply";
    ply_utils::export_to_ply(
        train_points_ply_file,
        data_loader_ptr->dataparser_ptr_->train_depth_pack_.xyz.cpu());
  }

  timer_cal_prior->toc_sum();
  return true;
}

void NeuralSLAM::batch_train() {
  if (k_prefilter > 0) {
    static auto p_prefilter_data_timer = llog::CreateTimer("prefilter_data");
    p_prefilter_data_timer->tic();
    prefilter_data(false);
    p_prefilter_data_timer->toc_sum();
  }
  static auto p_timer = llog::CreateTimer("build_occ_map");
  p_timer->tic();
  build_occ_map();

  torch::optim::AdamOptions adam_options;
  adam_options.lr(k_lr);
  adam_options.eps(1e-15);
  p_optimizer_ = std::make_shared<torch::optim::Adam>(
      local_map_ptr->parameters(), adam_options);
  p_timer->toc_sum();

  static auto p_train = llog::CreateTimer("train");
  p_train->tic();

  if (k_sdf_weight > 0) {
    nsdf_train(k_sdf_iter_step);
    export_checkpoint();
  }

  if (k_rgb_weight > 0) {
    auto train_color_num = data_loader_ptr->dataparser_ptr_->size(
        dataparser::DataType::TrainColor);
    int sample_step =
        data_loader_ptr->dataparser_ptr_->train_depth_pack_.xyz.size(0) / 1e5f;
    sample_step = std::max(sample_step, 1);
    auto init_pts = data_loader_ptr->dataparser_ptr_->train_depth_pack_.xyz
                        .slice(0, 0, -1, sample_step)
                        .cuda();
    neural_gs_ptr =
        std::make_shared<NeuralGS>(local_map_ptr, init_pts, train_color_num,
                                   0.5f * k_inner_map_size, k_sdf_weight > 0);

    // You should set gs_param_start_idx to enable densification work
    // normally.
    neural_gs_ptr->gs_param_start_idx = p_optimizer_->param_groups().size();
    for (auto &param_group : neural_gs_ptr->optimizer_params_groups_) {
      p_optimizer_->add_param_group(param_group);
    }

    // print out all optimizer parameters
    for (const auto &param_groups : p_optimizer_->param_groups()) {
      for (const auto &param : param_groups.params()) {
        std::cout << param.sizes() << "\n";
        std::cout << param.options() << "\n";
      }
    }
    gs_train(k_gs_iter_step);
  }

  p_train->toc_sum();
  llog::PrintLog();

  end();
  exit(0);
}

/* Exporter */
std::map<std::string, torch::Tensor>
NeuralSLAM::render_image(const torch::Tensor &_pose, const float &_scale,
                         const bool &training) {
  std::map<std::string, torch::Tensor> render_results;
  render_results = neural_gs_ptr->render(
      _pose, data_loader_ptr->dataparser_ptr_->sensor_.camera, false,
      k_bck_color);

  return render_results;
}

void NeuralSLAM::create_dir(const std::filesystem::path &base_path,
                            std::filesystem::path &color_path,
                            std::filesystem::path &depth_path,
                            std::filesystem::path &gt_color_path,
                            std::filesystem::path &render_color_path,
                            std::filesystem::path &gt_depth_path,
                            std::filesystem::path &render_depth_path,
                            std::filesystem::path &acc_path) {
  color_path = base_path / "color";
  depth_path = base_path / "depth";
  gt_color_path = color_path / "gt";
  std::filesystem::create_directories(gt_color_path);
  render_color_path = color_path / "renders";
  std::filesystem::create_directories(render_color_path);
  gt_depth_path = depth_path / "gt";
  std::filesystem::create_directories(gt_depth_path);
  render_depth_path = depth_path / "renders";
  std::filesystem::create_directories(render_depth_path);
  acc_path = base_path / "acc";
  std::filesystem::create_directories(acc_path);
}

void NeuralSLAM::render_path(bool eval, const int &fps, const bool &save) {
  // Early exit if evaluation requested with no evaluation data
  if (eval && (data_loader_ptr->dataparser_ptr_->size(
                   dataparser::DataType::EvalColor) == 0)) {
    return;
  }

  torch::GradMode::set_enabled(false);
  c10::cuda::CUDACachingAllocator::emptyCache();

  auto log_filename = log_file_.filename().replace_extension();
  auto log_path = k_output_path / log_filename;

  // Determine image type and create necessary directories once
  int image_type;
  std::string image_type_name;
  std::filesystem::path output_dir;

  if (!eval) {
    image_type = dataparser::DataType::RawColor;
    image_type_name = "train";
    output_dir = log_path / "train";
  } else {
    image_type = dataparser::DataType::EvalColor;
    image_type_name = "eval";
    output_dir = log_path / "eval";
  }

  // Only create directories if saving is enabled
  if (save) {
    auto color_path = output_dir / "color";
    auto depth_path = output_dir / "depth";
    auto normal_path = output_dir / "normal";
    auto gt_color_path = color_path / "gt";
    auto render_color_path = color_path / "renders";
    auto gt_depth_path = depth_path / "gt";
    auto render_depth_path = depth_path / "renders";

    // Create all directories at once instead of in create_dir function
    std::vector<std::filesystem::path> paths = {
        gt_color_path, render_color_path, gt_depth_path, render_depth_path,
        normal_path};
    for (const auto &path : paths) {
      std::filesystem::create_directories(path);
    }

    // If not eval, create test directories too
    if (!eval) {
      std::filesystem::path test_dir = log_path / "test";
      for (const auto &subdir : {"color/gt", "color/renders", "depth/gt",
                                 "depth/renders", "normal"}) {
        std::filesystem::create_directories(test_dir / subdir);
      }
    }
  }

  // Get image dimensions
  auto width = data_loader_ptr->dataparser_ptr_->sensor_.camera.width;
  auto height = data_loader_ptr->dataparser_ptr_->sensor_.camera.height;

  // Pre-allocate reusable buffers for image processing
  cv::Mat cat_color, cat_depth, depth_colormap;

  // Prepare a queue for background image writing
  struct ImageSaveTask {
    unsigned long index;
    int image_type;
    std::filesystem::path base_dir;
    std::map<std::string, torch::Tensor> renders;
  };
  std::vector<ImageSaveTask> save_queue;
  int reserve_size = 16;
  save_queue.reserve(reserve_size); // Pre-allocate to avoid reallocations

  // Periodically flush save queue to avoid excessive memory usage
  auto flush_save_queue = [&save_queue, this]() {
#pragma omp parallel for
    for (auto &task : save_queue) {
      // gt color
      auto gt_file = data_loader_ptr->dataparser_ptr_->get_file(
          task.index, task.image_type);
      auto gt_file_name = gt_file.filename();
      auto gt_color_path = task.base_dir / "color/gt" / gt_file_name;
      if (data_loader_ptr->dataparser_ptr_->sensor_.camera.scale != 1.0f) {
        auto gt = data_loader_ptr->dataparser_ptr_->get_image_cv_mat(
            task.index, task.image_type);
        cv::resize(
            gt, gt,
            cv::Size(data_loader_ptr->dataparser_ptr_->sensor_.camera.width,
                     data_loader_ptr->dataparser_ptr_->sensor_.camera.height));
        cv::imwrite(gt_color_path, gt);
      } else if (!std::filesystem::exists(gt_color_path)) {
        // ln gt_file to gt_color_path
        // get gt_file's absolute path
        gt_file = std::filesystem::absolute(gt_file);
        std::filesystem::create_symlink(gt_file, gt_color_path);
      }
      // render color
      auto render_file = task.base_dir / "color/renders" / gt_file_name;
      auto cv_render =
          utils::tensor_to_cv_mat(task.renders["color"].clamp(0.0f, 1.0f));
      cv::imwrite(render_file, cv_render);

      // render depth
      if (task.renders.find("depth") != task.renders.end()) {
        render_file = task.base_dir / "depth/renders" / gt_file_name;
        cv_render = utils::tensor_to_cv_mat(task.renders["depth"]);
        cv::imwrite(render_file, utils::apply_colormap_to_depth(cv_render));
      }
      // render normal
      if (task.renders.find("render_normal") != task.renders.end()) {
        render_file = task.base_dir / "normal" / gt_file_name;
        cv_render = utils::tensor_to_cv_mat(
            task.renders["render_normal"][0] * 0.5f + 0.5f);
        cv::imwrite(render_file, cv_render);
      }
    }
    save_queue.clear();
  };

  auto iter_bar =
      tq::trange(data_loader_ptr->dataparser_ptr_->size(image_type));
  iter_bar.set_prefix("Rendering");
  static auto p_timer_render = llog::CreateTimer("    render_train_image");

  // LLF skip flag - compute once rather than checking in loop
  bool llff_skip_enabled = (k_prefilter > 0) && k_llff;

  for (const auto &i : iter_bar) {
    // Skip frames if necessary (computed once for efficiency)
    if (llff_skip_enabled && (i % 8 != 0)) {
      continue;
    }

    // Render the image
    p_timer_render->tic();
    auto pose = data_loader_ptr->dataparser_ptr_->get_pose(i, image_type)
                    .slice(0, 0, 3);
    auto render_results = render_image(pose, 1.0, false);
    p_timer_render->toc_sum();

    // Skip processing if render failed or saving disabled
    if (render_results.empty() || !save) {
      continue;
    }

    // Queue images for saving - use test path for certain frames if needed
    bool is_test = (!eval) && k_llff && (i % 8 == 0);
    std::filesystem::path base_dir = is_test ? (log_path / "test") : output_dir;

    // Queue image saving tasks
    save_queue.push_back(
        {i, image_type, base_dir,
         std::map<std::string, torch::Tensor>{
             {"color", render_results["color"]},
             {"depth", render_results["depth"]},
             {"render_normal", render_results["render_normal"]}}});
    // Flush save queue periodically to avoid excessive memory usage
    if (save_queue.size() >= reserve_size) {
      flush_save_queue();
    }
  }

  // Final flush of save queue
  flush_save_queue();
}

void NeuralSLAM::render_path(std::string pose_file, const int &fps) {
  torch::NoGradGuard no_grad;
  std::cout << "Rendering pose_file: " << pose_file << std::endl;
  auto poses = std::get<0>(
      data_loader_ptr->dataparser_ptr_->load_poses(pose_file, false, 0));
  if (poses.size(0) == 0) {
    std::cout << "poses is empty" << std::endl;
    return;
  }
  c10::cuda::CUDACachingAllocator::emptyCache();

  std::filesystem::path color_path, depth_path, gt_color_path,
      render_color_path, gt_depth_path, render_depth_path, acc_path;

  auto log_filename = log_file_.filename().replace_extension();
  auto log_path = k_output_path / log_filename;

  int image_type;
  std::string image_type_name;
  image_type = 0;
  image_type_name = "path";
  create_dir(log_path / "path", color_path, depth_path, gt_color_path,
             render_color_path, gt_depth_path, render_depth_path, acc_path);

  auto width = data_loader_ptr->dataparser_ptr_->sensor_.camera.width;
  auto height = data_loader_ptr->dataparser_ptr_->sensor_.camera.height;

  auto video_format = cv::VideoWriter::fourcc('a', 'v', 'c', '1');
  cv::VideoWriter video_color(log_path /
                                  (image_type_name + "/render_color.mp4"),
                              video_format, fps, cv::Size(width, height));
  cv::VideoWriter video_depth;

  auto depth_scale = 1.0 / data_loader_ptr->dataparser_ptr_->depth_scale_inv_;
  auto iter_bar = tq::trange(poses.size(0));
  iter_bar.set_prefix("Rendering");
  static auto p_timer_render = llog::CreateTimer("    render_train_image");
  for (const auto &i : iter_bar) {
    p_timer_render->tic();
    auto render_results = render_image(poses[i].slice(0, 0, 3), 1.0, false);
    p_timer_render->toc_sum();
    if (!render_results.empty()) {
      auto render_color = utils::tensor_to_cv_mat(render_results["color"]);
      video_color.write(render_color);

      auto render_depth = utils::tensor_to_cv_mat(render_results["depth"]);
      cv::Mat depth_colormap = utils::apply_colormap_to_depth(render_depth);

      if (!video_depth.isOpened()) {
        video_depth =
            cv::VideoWriter(log_path / (image_type_name + "/render_depth.mp4"),
                            video_format, fps, cv::Size(width, height));
      }
      video_depth.write(depth_colormap);

      std::filesystem::path color_file_name = to_string(i) + ".png";
      cv::imwrite(render_color_path / color_file_name, render_color);
      cv::imwrite(render_depth_path / color_file_name, depth_colormap);
    }
  }
  video_color.release();
  video_depth.release();
}

float NeuralSLAM::export_test_image(bool is_eval, int idx,
                                    const std::string &prefix) {
  torch::NoGradGuard no_grad;

  int image_type = dataparser::DataType::TrainColor;
  std::string suffix = "train";
  if (is_eval) {
    if (data_loader_ptr->dataparser_ptr_->eval_color_poses_.numel() == 0) {
      image_type = dataparser::DataType::TestColor;
      suffix = "test";
    } else {

      image_type = dataparser::DataType::EvalColor;
      suffix = "eval";
    }
  }
  if (data_loader_ptr->dataparser_ptr_->size(image_type) <= idx) {
    idx = 0;
  }

  auto log_filename = log_file_.filename().replace_extension();
  auto log_path = k_output_path / log_filename;

  std::filesystem::path color_path = log_path / ("mid/color_" + suffix);
  std::filesystem::path depth_path = log_path / ("mid/depth_" + suffix);
  std::filesystem::path normal_path = log_path / ("mid/normal_" + suffix);
  std::filesystem::create_directories(normal_path);
  std::filesystem::path gt_color_path = color_path / "gt";
  std::filesystem::create_directories(gt_color_path);
  std::filesystem::path render_color_path = color_path / "renders";
  std::filesystem::create_directories(render_color_path);
  std::filesystem::path gt_depth_path = depth_path / "gt";
  std::filesystem::create_directories(gt_depth_path);
  std::filesystem::path render_depth_path = depth_path / "renders";
  std::filesystem::create_directories(render_depth_path);

  if (idx < 0) {
    idx = std::rand() % data_loader_ptr->dataparser_ptr_->size(0);
  }

  auto pose = data_loader_ptr->dataparser_ptr_->get_pose(idx, image_type)
                  .slice(0, 0, 3)
                  .to(k_device);
  auto render_results = render_image(pose, 1.0, false);

  float psnr = 0.0f;
  if (!render_results.empty()) {
    auto render_color = render_results["color"].clamp(0.0f, 1.0f);
    auto render_color_mat = utils::tensor_to_cv_mat(render_color);

    auto gt_color =
        data_loader_ptr->dataparser_ptr_->get_image(idx, image_type);
    auto render_depth = render_results["depth"];
    auto cv_render_depth = utils::tensor_to_cv_mat(render_depth);
    cv::Mat depth_colormap = utils::apply_colormap_to_depth(cv_render_depth);

    auto gt_color_file_name =
        data_loader_ptr->dataparser_ptr_->get_file(idx, image_type)
            .filename()
            .replace_extension(".png")
            .string();
    if (!std::filesystem::exists(gt_color_path / gt_color_file_name)) {
      auto gt_color_mat = utils::tensor_to_cv_mat(gt_color);
      cv::imwrite(gt_color_path / gt_color_file_name, gt_color_mat);
    }
    auto output_color_file_name = prefix + gt_color_file_name;
    cv::imwrite(render_color_path / output_color_file_name, render_color_mat);

    std::filesystem::path gt_depth_file =
        data_loader_ptr->dataparser_ptr_->get_file(idx, image_type + 1);
    auto gt_depth_file_name = gt_depth_file.filename().string();
    if (!std::filesystem::exists(gt_depth_path / gt_depth_file_name)) {
      auto gt_depth = data_loader_ptr->dataparser_ptr_->get_image_cv_mat(
          idx, image_type + 1);
      if (!gt_depth.empty()) {
        gt_depth = utils::apply_colormap_to_depth(gt_depth);
      }
      if (!gt_depth.empty()) {
        cv::imwrite(gt_depth_path / gt_depth_file_name, gt_depth);
      }
    }
    cv::imwrite(render_depth_path / output_color_file_name, depth_colormap);

    /* 2DGS */
    if (render_results["render_normal"].defined()) {
      auto render_normal = utils::tensor_to_cv_mat(
          render_results["render_normal"][0] * 0.5f + 0.5f);
      cv::imwrite(normal_path / output_color_file_name, render_normal);
    }
    if (neural_gs_ptr) {
      if (k_render_normal_weight > 0) {
        torch::Tensor depth2normal;
        if (k_depth_type == 0) {
          depth2normal = sensor::depth_to_normal(
              data_loader_ptr->dataparser_ptr_->sensor_.camera, pose,
              render_depth);
        } else {
          depth2normal = sensor::depth_to_normal(
              data_loader_ptr->dataparser_ptr_->sensor_.camera, pose,
              render_results["render_median"][0]);
        }
        depth2normal = depth2normal * render_results["alpha"][0].detach();
        auto depth2normal_cv =
            utils::tensor_to_cv_mat(depth2normal * 0.5f + 0.5f);

        std::filesystem::path depth_to_normal_path =
            log_path / ("mid/depth2normal_" + suffix);
        std::filesystem::create_directories(depth_to_normal_path);
        cv::imwrite(depth_to_normal_path / output_color_file_name,
                    depth2normal_cv);
      }
    }

    // eval color
    c10::cuda::CUDACachingAllocator::emptyCache();
    auto eval_python_cmd =
        "python3 " + k_package_path.string() +
        "/eval/image_metrics/metrics_single.py --gt_color " +
        (gt_color_path / gt_color_file_name).string() + " --renders_color " +
        (render_color_path / output_color_file_name).string();
    std::cout << "\n\033[34mConducting rendering evaluation command: "
              << eval_python_cmd << "\033[0m\n";
    int ret = std::system(eval_python_cmd.c_str());

    psnr = loss_utils::psnr(gt_color.to(render_color.device()).unsqueeze(0),
                            render_color.unsqueeze(0));
  }

  return psnr;
}

// https://pytorch.org/tutorials/advanced/cpp_frontend.html#checkpointing-and-recovering-the-training-state
void NeuralSLAM::export_checkpoint() {
  if (!k_output_path.empty()) {
    auto save_path = k_model_path / "local_map_checkpoint.pt";
    torch::save(local_map_ptr, save_path);
    write_pt_params();

    if (neural_gs_ptr) {
      auto gs_ply_save_path = k_model_path / "gs.ply";
      neural_gs_ptr->export_gs_to_ply(gs_ply_save_path);
    }
  }
}

void NeuralSLAM::load_checkpoint(
    const std::filesystem::path &_checkpoint_path) {
  auto pt_path = _checkpoint_path / "local_map_checkpoint.pt";
  if (std::filesystem::exists(pt_path)) {
    read_pt_params();
    local_map_ptr = std::make_shared<LocalMap>(
        k_map_origin, k_x_min, k_x_max, k_y_min, k_y_max, k_z_min, k_z_max);
    torch::load(local_map_ptr, pt_path);
    log_file_ = "gs_log.txt";
  } else {
    std::cout << "\033[31mCheckpoint file not found: " << pt_path
              << "\033[0m\n";
  }
  if (k_rgb_weight > 0) {
    // check if gs.ply exist
    auto gs_ply_path = _checkpoint_path / "gs.ply";
    if (std::filesystem::exists(gs_ply_path)) {
      // load gs.ply
      neural_gs_ptr = std::make_shared<NeuralGS>(local_map_ptr, gs_ply_path);
    } else {
      std::cout << "No gs.ply found in " << gs_ply_path
                << ", skip loading gs.\033[0m\n";
    }
  }

  auto as_prior_ply_file = _checkpoint_path / "as_occ_prior.ply";
  std::map<std::string, torch::Tensor> map_ply_tensor;
  if (ply_utils::read_ply_file_to_map_tensor(as_prior_ply_file, map_ply_tensor,
                                             torch::Device(torch::kCUDA))) {
    local_map_ptr->update_octree_as(map_ply_tensor["xyz"], true, 0);
  } else {
    throw std::runtime_error("Failed to load " + as_prior_ply_file.string());
  }
  std::cout << "Load checkpoint done!" << "\n";
}

void NeuralSLAM::save_mesh(const float &res, const bool &cull_mesh,
                           const std::string &prefix, const bool &return_mesh) {
  cout << "\033[1;34m\nStart meshing map...\n\033[0m";
#ifdef ENABLE_ROS
  std_msgs::Header header;
  header.frame_id = "world";
  header.stamp = ros::Time::now();
  local_map_ptr->meshing_(mesh_pub, mesh_color_pub, header, res, true);
#else
  local_map_ptr->meshing_(k_export_res, true);
#endif

  auto log_filename = log_file_.filename().replace_extension();
  auto log_path = k_output_path / log_filename;
  if (cull_mesh) {
    local_map_ptr->p_mesher_->save_mesh(
        log_path, k_vis_attribute, prefix + std::to_string(res), return_mesh,
        data_loader_ptr->dataparser_ptr_);
  } else {
    local_map_ptr->p_mesher_->save_mesh(
        log_path, k_vis_attribute, prefix + std::to_string(res), return_mesh);
  }
}

void NeuralSLAM::eval_mesh() {
  auto gt_mesh_path = data_loader_ptr->dataparser_ptr_->get_gt_mesh_path();
  if (std::filesystem::exists(gt_mesh_path)) {
    c10::cuda::CUDACachingAllocator::emptyCache();
    auto log_filename = log_file_.filename().replace_extension();
    auto log_path = k_output_path / log_filename;
    auto eval_python_cmd = "python3 " + k_package_path.string() +
                           "/eval/structure_metrics/evaluator.py --pred_mesh " +
                           log_path.string();
    if (k_cull_mesh) {
      eval_python_cmd += "/mesh_culled_" + std::to_string(k_export_res) +
                         ".ply --gt_pcd " +
                         data_loader_ptr->dataparser_ptr_->get_gt_mesh_path();
    } else {
      eval_python_cmd += "/mesh_" + std::to_string(k_export_res) +
                         ".ply --gt_pcd " +
                         data_loader_ptr->dataparser_ptr_->get_gt_mesh_path();
    }
    std::cout << "\033[34mConducting structure evaluation command: "
              << eval_python_cmd << "\033[0m\n";
    int ret = std::system(eval_python_cmd.c_str());
    std::cout << "\033[32mEvaluation finished. Please check the results in the "
                 "folder: "
              << log_path << "\033[0m\n";
  } else {
    std::cout
        << "\033[31mNo ground truth mesh found, skip evaluation.\033[0m\n";
    return;
  }
}

void NeuralSLAM::eval_render() {
  c10::cuda::CUDACachingAllocator::emptyCache();
  auto log_filename = log_file_.filename().replace_extension();
  auto log_path = k_output_path / log_filename;
  auto eval_python_cmd = "python3 " + k_package_path.string() +
                         "/eval/image_metrics/metrics.py -m " +
                         (log_path / "train/color").string();
  std::cout << "\n\033[34mConducting rendering evaluation command: "
            << eval_python_cmd << "\033[0m\n";
  int ret = std::system(eval_python_cmd.c_str());

  eval_python_cmd = "python3 " + k_package_path.string() +
                    "/eval/image_metrics/metrics.py -m " +
                    (log_path / "eval/color").string();
  std::cout << "\n\033[34mConducting rendering evaluation command: "
            << eval_python_cmd << "\033[0m\n";
  ret = std::system(eval_python_cmd.c_str());

  eval_python_cmd = "python3 " + k_package_path.string() +
                    "/eval/image_metrics/metrics.py -m " +
                    (log_path / "test/color").string();
  std::cout << "\n\033[34mConducting rendering evaluation command: "
            << eval_python_cmd << "\033[0m\n";
  ret = std::system(eval_python_cmd.c_str());
  std::cout << "\033[32mEvaluation finished. Please check the results in the "
               "folder: "
            << log_path << "\033[0m\n";
}

void NeuralSLAM::export_timing(bool print, const std::string &prefix) {
  if (print) {
    llog::PrintLog();
  }
  auto log_filename = log_file_.filename().replace_extension();
  auto log_path = k_output_path / log_filename;
  llog::SaveLog(log_path / (prefix + "timing.txt"));
}

void NeuralSLAM::plot_log(const std::string &log_file) {
  c10::cuda::CUDACachingAllocator::emptyCache();
  auto cmd = "python3 " + k_package_path.string() + "/eval/draw_loss.py -l " +
             (k_output_path / log_file).string();
  std::cout << "\033[34mConducting draw_loss command: " << cmd << "\033[0m\n";
  auto ret = std::system(cmd.c_str());
}
/* /Exporter */

void NeuralSLAM::keyboard_loop() {
  while (true) {
    std::string input;
    std::getline(std::cin, input);

    if (input.empty()) {
      continue;
    }

    char c = input[0];
    switch (c) {
    case 'm': {
      float res = k_export_res;
      if (input.length() > 2) {
        try {
          res = std::stof(input.substr(2));
        } catch (std::exception &e) {
          std::cout << "Invalid resolution value. Using default: "
                    << k_export_res << "\n";
        }
      }
      std::cout << "Exporting mesh with resolution: " << res << "\n";
      save_mesh(res, k_cull_mesh);
      break;
    }
    case 'e': {
      eval_mesh();
      eval_render();
      break;
    }
    case 'q': {
      end();
      break;
    }
    case 'r': {
      c10::cuda::CUDACachingAllocator::emptyCache();
      render_path(false, k_fps);
      render_path(true, 2);
      eval_render();
      break;
    }
    case 'p': {
      save_image = true;
      break;
    }
    case 'o': {
      export_checkpoint();
      break;
    }
    case 'i': {
      // if k_dataset_path is a file
      std::filesystem::path dataset_path = k_dataset_path;
      if (std::filesystem::is_regular_file(k_dataset_path)) {
        dataset_path = k_dataset_path.parent_path();
      }
      if (!std::filesystem::exists(dataset_path / "color_poses.txt")) {
        for (int i = 0; i < data_loader_ptr->dataparser_ptr_->size(
                                dataparser::DataType::RawColor);
             i++) {
          auto pose = data_loader_ptr->dataparser_ptr_->get_pose(i, 0);
          std::ofstream ofs(dataset_path / "color_poses.txt", std::ios::app);
          for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
              ofs << pose[i][j].item<float>() << " ";
            }
            ofs << "\n";
          }
        }
      }

      auto eval_python_cmd = "python3 " + k_package_path.string() +
                             "/eval/inter_poses.py --data_dir " +
                             dataset_path.string() +
                             " --key_poses skip --skip 50 --n_out_poses 500";
      std::cout << "\033[34mInterpolating Path Generation command: "
                << eval_python_cmd << "\033[0m\n";
      int ret = std::system(eval_python_cmd.c_str());

      std::string pose_file = dataset_path / "inter_color_poses.txt";
      render_path(pose_file, k_fps);
      break;
    }
    case 'u': {
      std::string pose_file =
          "/home/chrisliu/Projects/rimv2_ws/src/RIM2/data/"
          "FAST_LIVO2_RIM_Datasets/culture01/inter_color_poses.txt";
      render_path(pose_file, k_fps);
      break;
    }
    case 'v': {
      render_path(false, k_fps, false);
      llog::PrintLog();
      break;
    }
    case 'd': {
      std::cout << "Double render resolution! From "
                << data_loader_ptr->dataparser_ptr_->sensor_.camera.width << "x"
                << data_loader_ptr->dataparser_ptr_->sensor_.camera.height
                << " to ";
      data_loader_ptr->dataparser_ptr_->sensor_.camera.width *= 2;
      data_loader_ptr->dataparser_ptr_->sensor_.camera.height *= 2;
      data_loader_ptr->dataparser_ptr_->sensor_.camera.fx *= 2;
      data_loader_ptr->dataparser_ptr_->sensor_.camera.fy *= 2;
      data_loader_ptr->dataparser_ptr_->sensor_.camera.cx *= 2;
      data_loader_ptr->dataparser_ptr_->sensor_.camera.cy *= 2;
      std::cout << data_loader_ptr->dataparser_ptr_->sensor_.camera.width << "x"
                << data_loader_ptr->dataparser_ptr_->sensor_.camera.height
                << "\n";
      break;
    }
    case 's': {
      std::cout << "Dedouble render resolution! From "
                << data_loader_ptr->dataparser_ptr_->sensor_.camera.width << "x"
                << data_loader_ptr->dataparser_ptr_->sensor_.camera.height
                << " to ";
      data_loader_ptr->dataparser_ptr_->sensor_.camera.width /= 2;
      data_loader_ptr->dataparser_ptr_->sensor_.camera.height /= 2;
      data_loader_ptr->dataparser_ptr_->sensor_.camera.fx /= 2;
      data_loader_ptr->dataparser_ptr_->sensor_.camera.fy /= 2;
      data_loader_ptr->dataparser_ptr_->sensor_.camera.cx /= 2;
      data_loader_ptr->dataparser_ptr_->sensor_.camera.cy /= 2;
      std::cout << data_loader_ptr->dataparser_ptr_->sensor_.camera.width << "x"
                << data_loader_ptr->dataparser_ptr_->sensor_.camera.height
                << "\n";
      break;
    }
    case 'f': {
      std::cout << "Double render resolution! From "
                << data_loader_ptr->dataparser_ptr_->sensor_.camera.width << "x"
                << data_loader_ptr->dataparser_ptr_->sensor_.camera.height
                << " to ";
      data_loader_ptr->dataparser_ptr_->sensor_.camera.width *= 2;
      data_loader_ptr->dataparser_ptr_->sensor_.camera.height *= 2;
      std::cout << data_loader_ptr->dataparser_ptr_->sensor_.camera.width << "x"
                << data_loader_ptr->dataparser_ptr_->sensor_.camera.height
                << "\n";
      break;
    }
    case 'g': {
      std::cout << "Dedouble render resolution! From "
                << data_loader_ptr->dataparser_ptr_->sensor_.camera.width << "x"
                << data_loader_ptr->dataparser_ptr_->sensor_.camera.height
                << " to ";
      data_loader_ptr->dataparser_ptr_->sensor_.camera.width /= 2;
      data_loader_ptr->dataparser_ptr_->sensor_.camera.height /= 2;
      std::cout << data_loader_ptr->dataparser_ptr_->sensor_.camera.width << "x"
                << data_loader_ptr->dataparser_ptr_->sensor_.camera.height
                << "\n";
      break;
    }
    case 'n': {
      std::cout << "Increase k_near from " << k_near << " to ";
      k_near *= 1.2f;
      std::cout << k_near << "\n";
      break;
    }
    case 'b': {
      std::cout << "Decrease k_near from " << k_near << " to ";
      k_near *= 0.8f;
      std::cout << k_near << "\n";
      break;
    }
    case 'k': {
      std::cout << "Double width! From "
                << data_loader_ptr->dataparser_ptr_->sensor_.camera.width
                << " to ";
      data_loader_ptr->dataparser_ptr_->sensor_.camera.width *= 2;
      data_loader_ptr->dataparser_ptr_->sensor_.camera.cx *= 2;
      std::cout << data_loader_ptr->dataparser_ptr_->sensor_.camera.width
                << "\n";
      break;
    }
    case 'j': {
      std::cout << "DeDouble width! From "
                << data_loader_ptr->dataparser_ptr_->sensor_.camera.width
                << " to ";
      data_loader_ptr->dataparser_ptr_->sensor_.camera.width /= 2;
      data_loader_ptr->dataparser_ptr_->sensor_.camera.cx /= 2;
      std::cout << data_loader_ptr->dataparser_ptr_->sensor_.camera.width
                << "\n";
      break;
    }
    default: {
      std::cout << "NEURAL MAPPING KEYBOARD COMMANDS:\n";
      std::cout << "m [res]: save mesh (optional resolution parameter)\n";
      std::cout << "g: save inrange pointcloud\n";
      std::cout << "e: eval mesh and render\n";
      std::cout << "q: end\n";
      std::cout << "r: render eval\n";
      std::cout << "p: save snapshot\n";
      std::cout << "o: save check point\n";
      std::cout << "i: render interpolated path\n";
      std::cout << "t: toggle render type\n";
      std::cout << "d: double render resolution\n";
      std::cout << "s: dedouble render resolution\n";
      std::cout << "v: test render speed\n";
    }
    }

    std::chrono::milliseconds dura(100);
    std::this_thread::sleep_for(dura);
  }
}

bool NeuralSLAM::end() {
  export_timing(true, log_file_.filename());
  export_checkpoint();
  if ((k_sdf_weight > 0) && (k_export_mesh)) {
    // clear VRAM
    c10::cuda::CUDACachingAllocator::emptyCache();
    save_mesh(k_export_res, k_cull_mesh);
    eval_mesh();
  }
  if (neural_gs_ptr) {
    c10::cuda::CUDACachingAllocator::emptyCache();
    render_path(false, k_fps);
    render_path(true, 2);
    eval_render();
  }

  return true;
}

void NeuralSLAM::misc_loop() {
  while (true) {
    if (misc_trigger >= 0) {
      plot_log(log_file_);
      export_timing(false, log_file_.filename());
      misc_trigger = -1;
    }
    std::chrono::milliseconds dura(100);
    std::this_thread::sleep_for(dura);
  }
}

#ifdef ENABLE_ROS
void NeuralSLAM::pretrain_loop() {
  while (true) {
    visualization();

    static std::chrono::milliseconds dura(10);
    std::this_thread::sleep_for(dura);
  }
}

NeuralSLAM::NeuralSLAM(ros::NodeHandle &_nh, const int &mode,
                       const std::filesystem::path &_config_path,
                       const std::filesystem::path &_data_path)
    : NeuralSLAM(mode, _config_path, _data_path) {
  register_subscriber(_nh);
  register_publisher(_nh);
}

void NeuralSLAM::register_subscriber(ros::NodeHandle &nh) {
  rviz_pose_sub = nh.subscribe("/rviz/current_camera_pose", 1,
                               &NeuralSLAM::rviz_pose_callback, this);
}

void NeuralSLAM::register_publisher(ros::NodeHandle &nh) {
  pose_pub = nh.advertise<geometry_msgs::PoseStamped>("pose", 1);
  path_pub = nh.advertise<nav_msgs::Path>("path", 1);
  odom_pub = nh.advertise<nav_msgs::Odometry>("odom", 1);

  mesh_pub = nh.advertise<mesh_msgs::MeshGeometryStamped>("mesh", 1);
  mesh_color_pub =
      nh.advertise<mesh_msgs::MeshVertexColorsStamped>("mesh_color", 1);
  vis_shift_map_pub =
      nh.advertise<visualization_msgs::Marker>("vis_shift_map", 1);
  pointcloud_pub = nh.advertise<sensor_msgs::PointCloud2>("pointcloud", 1);
  rgb_pub = nh.advertise<sensor_msgs::Image>("rgb", 1);
  depth_pub = nh.advertise<sensor_msgs::Image>("depth", 1);
  // wait all publisher to be ready
  ros::Duration(0.5).sleep();
}

void NeuralSLAM::rviz_pose_callback(
    const geometry_msgs::PoseConstPtr &_rviz_pose_ptr) {
  auto pos =
      torch::tensor({_rviz_pose_ptr->position.x, _rviz_pose_ptr->position.y,
                     _rviz_pose_ptr->position.z},
                    torch::kFloat);
  auto quat = torch::tensor(
      {_rviz_pose_ptr->orientation.w, _rviz_pose_ptr->orientation.x,
       _rviz_pose_ptr->orientation.y, _rviz_pose_ptr->orientation.z},
      torch::kFloat);
  auto rot = utils::quat_to_rot(quat).matmul(
      coords::opencv_to_opengl_camera_rotation());
  rviz_pose_ = torch::cat({rot, pos.view({3, 1})}, 1);
}

/* Publisher */
void NeuralSLAM::pub_pose(const torch::Tensor &_pose,
                          std_msgs::Header _header) {
  if (pose_pub.getNumSubscribers() > 0 || path_pub.getNumSubscribers() > 0) {
    if (_header.frame_id.empty()) {
      _header.frame_id = "world";
      _header.stamp = ros::Time::now();
    }
    auto pos = _pose.slice(1, 3, 4).detach().cpu();
    auto quat = utils::rot_to_quat(_pose.slice(1, 0, 3).detach().cpu());
    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.header = _header;
    pose_stamped.pose.position.x = pos[0].item<float>();
    pose_stamped.pose.position.y = pos[1].item<float>();
    pose_stamped.pose.position.z = pos[2].item<float>();
    pose_stamped.pose.orientation.w = quat[0].item<float>();
    pose_stamped.pose.orientation.x = quat[1].item<float>();
    pose_stamped.pose.orientation.y = quat[2].item<float>();
    pose_stamped.pose.orientation.z = quat[3].item<float>();
    pose_pub.publish(pose_stamped);

    path_msg.header = _header;
    path_msg.poses.emplace_back(pose_stamped);
    path_pub.publish(path_msg);

    static tf::TransformBroadcaster br;
    tf::Transform transform;
    tf::Quaternion q;
    // body frame
    transform.setOrigin(tf::Vector3(pose_stamped.pose.position.x,
                                    pose_stamped.pose.position.y,
                                    pose_stamped.pose.position.z));
    q.setW(pose_stamped.pose.orientation.w);
    q.setX(pose_stamped.pose.orientation.x);
    q.setY(pose_stamped.pose.orientation.y);
    q.setZ(pose_stamped.pose.orientation.z);
    transform.setRotation(q);
    br.sendTransform(
        tf::StampedTransform(transform, _header.stamp, "world", "depth"));
    transform.setIdentity();
    br.sendTransform(tf::StampedTransform(transform, _header.stamp, "world",
                                          _header.frame_id));
  }
}

void NeuralSLAM::pub_path(std_msgs::Header _header) {
  if (path_pub.getNumSubscribers() > 0) {
    if (_header.frame_id.empty()) {
      _header.frame_id = "world";
      _header.stamp = ros::Time::now();
    }
    for (int i = 0; i < data_loader_ptr->dataparser_ptr_->size(
                            dataparser::DataType::TrainColor);
         ++i) {
      auto _pose = data_loader_ptr->dataparser_ptr_->get_pose(
          i, dataparser::DataType::TrainColor);
      auto pos = _pose.slice(1, 3, 4).detach().cpu();
      auto quat = utils::rot_to_quat(_pose.slice(1, 0, 3).detach().cpu());
      geometry_msgs::PoseStamped pose_stamped;
      pose_stamped.header = _header;
      pose_stamped.pose.position.x = pos[0].item<float>();
      pose_stamped.pose.position.y = pos[1].item<float>();
      pose_stamped.pose.position.z = pos[2].item<float>();
      pose_stamped.pose.orientation.w = quat[0].item<float>();
      pose_stamped.pose.orientation.x = quat[1].item<float>();
      pose_stamped.pose.orientation.y = quat[2].item<float>();
      pose_stamped.pose.orientation.z = quat[3].item<float>();
      nav_msgs::Odometry odom_msg;
      odom_msg.header = _header;
      odom_msg.pose.pose = pose_stamped.pose;
      odom_pub.publish(odom_msg);

      path_msg.header = _header;
      path_msg.poses.emplace_back(pose_stamped);
    }
    path_pub.publish(path_msg);
  }
}

void NeuralSLAM::pub_pointcloud(const ros::Publisher &_pub,
                                const torch::Tensor &_xyz,
                                std_msgs::Header _header) {
  if (_pub.getNumSubscribers() > 0) {
    if (_header.frame_id.empty()) {
      _header.frame_id = "world";
      _header.stamp = ros::Time::now();
    }
    sensor_msgs::PointCloud2 pcl_msg;
    pcl::toROSMsg(utils::tensor_to_pointcloud(_xyz), pcl_msg);
    pcl_msg.header = _header;
    _pub.publish(pcl_msg);
  }
}

void NeuralSLAM::pub_pointcloud(const ros::Publisher &_pub,
                                const torch::Tensor &_xyz,
                                const torch::Tensor &_rgb,
                                std_msgs::Header _header) {
  if (_pub.getNumSubscribers() > 0) {
    if (_header.frame_id.empty()) {
      _header.frame_id = "world";
      _header.stamp = ros::Time::now();
    }
    sensor_msgs::PointCloud2 pcl_msg =
        utils::tensor_to_pointcloud_msg(_xyz, _rgb);
    pcl_msg.header = _header;
    _pub.publish(pcl_msg);
  }
}

void NeuralSLAM::pub_image(const ros::Publisher &_pub,
                           const torch::Tensor &_image,
                           std_msgs::Header _header) {
  if (_pub.getNumSubscribers() > 0) {
    if (_header.frame_id.empty()) {
      _header.frame_id = "world";
      _header.stamp = ros::Time::now();
    }
    sensor_msgs::Image img_msg = utils::tensor_to_img_msg(_image);
    img_msg.header = _header;
    _pub.publish(img_msg);
  }
}

void NeuralSLAM::pub_render_image(std_msgs::Header _header) {
  if (!neural_gs_ptr) {
    return;
  }
  if ((rviz_pose_.numel() > 0) &&
      (rgb_pub.getNumSubscribers() > 0 || depth_pub.getNumSubscribers() > 0)) {
    if (_header.frame_id.empty()) {
      _header.frame_id = "world";
      _header.stamp = ros::Time::now();
    }

    torch::NoGradGuard no_grad;
    auto render_results = render_image(rviz_pose_, 1.0, false);

    if (!render_results.empty()) {
      if (rgb_pub.getNumSubscribers() > 0) {
        pub_image(rgb_pub, render_results["color"], _header);
      }
      if (depth_pub.getNumSubscribers() > 0) {
        pub_image(depth_pub, render_results["depth"], _header);
      }
      if (save_image) {
        auto log_filename = log_file_.filename().replace_extension();
        auto log_path = k_output_path / log_filename;
        static std::filesystem::path snapshot_color_path =
            log_path / "snapshot/color";
        if (!std::filesystem::exists(snapshot_color_path)) {
          std::filesystem::create_directories(snapshot_color_path);
        }
        static std::filesystem::path snapshot_depth_path =
            log_path / "snapshot/depth";
        if (!std::filesystem::exists(snapshot_depth_path)) {
          std::filesystem::create_directories(snapshot_depth_path);
        }
        static int count = 0;
        cv::imwrite(snapshot_color_path / (to_string(count) + ".png"),
                    utils::tensor_to_cv_mat(render_results[0]));
        cv::imwrite(snapshot_depth_path / (to_string(count) + ".png"),
                    utils::apply_colormap_to_depth(
                        utils::tensor_to_cv_mat(render_results["depth"])));
        // save pose
        std::ofstream pose_file(snapshot_color_path /
                                (to_string(count) + ".txt"));
        pose_file << rviz_pose_ << std::endl;

        count++;
        save_image = false;
        cout << "save snapshot in " << snapshot_color_path << endl;
        cout << "Snapshot pose:\n" << rviz_pose_ << endl;
      }
    }
  }
}

void NeuralSLAM::visualization(const torch::Tensor &_xyz,
                               std_msgs::Header _header) {
  torch::NoGradGuard no_grad;
  if (_header.frame_id.empty()) {
    _header.frame_id = "world";
    _header.stamp = ros::Time::now();
  }
  static auto p_timer = llog::CreateTimer("visualization");
  p_timer->tic();

  pub_render_image(_header);
  pub_path(_header);

  if (mesh_pub.getNumSubscribers() > 0) {
    local_map_ptr->meshing_(mesh_pub, mesh_color_pub, _header, k_vis_res,
                            false);
  }

  if (vis_shift_map_pub.getNumSubscribers() > 0) {
    auto vis_shift_map =
        utils::get_vis_shift_map(local_map_ptr->pos_W_M_, k_x_min, k_x_max,
                                 k_y_min, k_y_max, k_z_min, k_z_max);
    if (!vis_shift_map.points.empty()) {
      vis_shift_map.header = _header;
      vis_shift_map_pub.publish(vis_shift_map);
    }
  }
  p_timer->toc_sum();
}
/* /Publisher */
#endif