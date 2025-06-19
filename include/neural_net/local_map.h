#pragma once

#include "mesher/mesher.h"
#include "neural_net/encoding_map.h"
#include "tcnn_binding/tcnn_binding.h"
#include "utils/ray_utils/ray_utils.h"

#ifdef ENABLE_ROS
#include "ros/publisher.h"
#endif
struct LocalMap : SubMap {
  typedef std::shared_ptr<LocalMap> Ptr;
  explicit LocalMap(const torch::Tensor &_pos_W_M, float _x_min, float _x_max,
                    float _y_min, float _y_max, float _z_min, float _z_max);

  std::shared_ptr<mesher::Mesher> p_mesher_;

  std::shared_ptr<EncodingMap> p_encoding_map_;
  // Decoder
  torch::nn::Sequential decoder_;
  // tcnn
  std::shared_ptr<TCNNNetwork> p_decoder_tcnn_;
  std::vector<torch::optim::OptimizerParamGroup> optimizer_params_groups_;

  std::shared_ptr<TCNNEncoding> p_dir_encoder_tcnn_;
  torch::Tensor dir_mask;

  torch::Tensor partition_pos_W_M_;   // pos: Map(xyz) to World;
  torch::Tensor partition_map_index_; // index: Map to World; [3]

  EncodingMap *new_map(const torch::Tensor &_pos_W_M);

  void unfreeze_net();

  void freeze_net();

  torch::Tensor get_feat(const torch::Tensor &xyz, const int &encoding_type = 0,
                         const bool &normalized = false);

  std::vector<torch::Tensor> get_sdf(const torch::Tensor &xyz);
  std::vector<torch::Tensor> get_gradient(const torch::Tensor &_xyz,
                                          const float &delta = 0.01,
                                          torch::Tensor _sdf = torch::Tensor(),
                                          bool _heissian = false,
                                          bool _numerical_grad = true);

  std::vector<torch::Tensor> marching_cubes(ros::Publisher &mesh_pub,
                                            ros::Publisher &mesh_color_pub,
                                            std_msgs::Header &header,
                                            float _res, bool _save,
                                            const std::string &uuid);

  void meshing_(float _res, bool _save);

  DepthSamples sample(const DepthSamples &_samples, int voxel_sample_num = 1,
                      const bool &_sample_free = true);

  DepthSamples filter_sample(DepthSamples &_samples);
  void freeze_decoder();

#ifdef ENABLE_ROS
  static void pub_mesh(const ros::Publisher &_mesh_pub,
                       const ros::Publisher &_mesh_color_pub,
                       const torch::Tensor &vertice, const torch::Tensor &face,
                       const torch::Tensor &color,
                       const std_msgs::Header &_header,
                       const std::string &_uuid);

  void meshing_(ros::Publisher &mesh_pub, ros::Publisher &mesh_color_pub,
                std_msgs::Header &header, float _res, bool _save = false,

                const std::string &uuid = "mesh_map");
#endif
};