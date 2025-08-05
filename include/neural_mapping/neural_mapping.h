#pragma once

#include <Eigen/Core>
#ifdef ENABLE_ROS
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TransformStamped.h>
#include <ros/ros.h>
#endif

#include "data_loader/data_loader.h"
#include "neural_gaussian/neural_gaussian.h"
#include "neural_net/local_map.h"

class NeuralSLAM {
public:
  typedef std::shared_ptr<NeuralSLAM> Ptr;
  NeuralSLAM(const int &mode, const std::filesystem::path &_config_path,
             const std::filesystem::path &_data_path = "");

#ifdef ENABLE_ROS
  NeuralSLAM(ros::NodeHandle &nh, const int &mode,
             const std::filesystem::path &_config_path,
             const std::filesystem::path &_data_path = "");
#endif

  // private:
  NeuralGS::Ptr neural_gs_ptr;

  LocalMap::Ptr local_map_ptr;
  std::shared_ptr<torch::optim::Adam> p_optimizer_;

  dataloader::DataLoader::Ptr data_loader_ptr;

#ifdef ENABLE_ROS
  // ROS stuff
  ros::Subscriber rviz_pose_sub;
  ros::Publisher pose_pub, path_pub, odom_pub;
  ros::Publisher mesh_pub, mesh_color_pub, vis_shift_map_pub, pointcloud_pub,
      rgb_pub, depth_pub;

  nav_msgs::Path path_msg;

  torch::Tensor rviz_pose_ = torch::Tensor();

  void register_subscriber(ros::NodeHandle &nh);
  void register_publisher(ros::NodeHandle &nh);

  void rviz_pose_callback(const geometry_msgs::PoseConstPtr &_rviz_pose_ptr);

  void visualization(const torch::Tensor &_xyz = {},
                     std_msgs::Header _header = std_msgs::Header());
  void pub_pose(const torch::Tensor &_pose,
                std_msgs::Header _header = std_msgs::Header());
  void pub_path(std_msgs::Header _header = std_msgs::Header());
  static void pub_pointcloud(const ros::Publisher &_pub,
                             const torch::Tensor &_xyz,
                             std_msgs::Header _header = std_msgs::Header());
  static void pub_pointcloud(const ros::Publisher &_pub,
                             const torch::Tensor &_xyz,
                             const torch::Tensor &_rgb,
                             std_msgs::Header _header = std_msgs::Header());
  static void pub_image(const ros::Publisher &_pub, const torch::Tensor &_image,
                        std_msgs::Header _header = std_msgs::Header());

  void pub_render_image(std_msgs::Header _header);
  void pretrain_loop();
#endif

  // thread, buffer stuff
  std::thread mapper_thread, keyboard_thread, misc_thread;

  bool save_image = false;

  void prefilter_data(const bool &export_img = false);

  bool build_occ_map();

  DepthSamples sample(DepthSamples batch_ray_samples, const float &sample_std,
                      const bool &_sample_free = true);

  torch::Tensor sdf_regularization(const torch::Tensor &xyz,
                                   const torch::Tensor &sdf = torch::Tensor(),
                                   const bool &curvate_enable = false,
                                   const std::string &name = "");
  std::tuple<torch::Tensor, DepthSamples>
  sdf_train_batch_iter(const int &iter, const bool &_sample_free = true);
  std::tuple<torch::Tensor, std::map<std::string, torch::Tensor>>
  gs_train_batch_iter(const int &iter, const bool &opt_struct = true);

  void nsdf_train(int _opt_iter);
  void gs_train(int _opt_iter);

  void sdf_train_callback(const int &_opt_iter, const int &_total_iter,
                          const RaySamples &point_samples,
                          const bool &_update_lr = true);

  void keyboard_loop();
  void batch_train();
  void misc_loop();
  int misc_trigger = -1;
  std::filesystem::path log_file_;

  void create_dir(const std::filesystem::path &base_path,
                  std::filesystem::path &color_path,
                  std::filesystem::path &depth_path,
                  std::filesystem::path &gt_color_path,
                  std::filesystem::path &render_color_path,
                  std::filesystem::path &gt_depth_path,
                  std::filesystem::path &render_depth_path,
                  std::filesystem::path &acc_path);

  std::map<std::string, torch::Tensor> render_image(const int &img_idx,
                                                    const int &pose_type);
  std::map<std::string, torch::Tensor>
  render_image(const torch::Tensor &_pose, const float &_scale = 1.0,
               const bool &training = true);

  void render_path(bool eval, const int &fps = 30, const bool &save = true);
  void render_path(std::string pose_file, const int &fps = 30);

  float export_test_image(bool is_eval = false, int idx = -1,
                          const std::string &prefix = "");
  void export_checkpoint();

  void load_pretrained(const std::filesystem::path &_pretrained_path);
  void load_checkpoint(const std::filesystem::path &_checkpoint_path);

  void save_mesh(const float &res, const bool &cull_mesh = false,
                 const std::string &prefix = "",
                 const bool &return_mesh = false);
  void eval_mesh();
  void eval_render();
  static void plot_log(const std::string &log_file);
  void export_timing(bool print = false, const std::string &prefix = "");

  bool end();
};