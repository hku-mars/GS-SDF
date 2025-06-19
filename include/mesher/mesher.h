#pragma once

#include <torch/torch.h>
#include <vector>

#include "data_loader/data_loader.h"

#ifdef ENABLE_ROS
#include <mesh_msgs/MeshGeometryStamped.h>
#include <mesh_msgs/MeshVertexColorsStamped.h>
#endif

namespace mesher {
#ifdef ENABLE_ROS
void tensor_to_mesh(mesh_msgs::MeshGeometry &mesh,
                    mesh_msgs::MeshVertexColors &mesh_color,
                    const torch::Tensor &vertice, const torch::Tensor &face,
                    const torch::Tensor &color);
#endif

class Mesher {
public:
  Mesher();

  std::vector<torch::Tensor> vec_face_attr_, vec_face_, vec_vertice_;
  torch::Tensor faces_, vertices_;
  void save_mesh(const std::string &output_path, bool vis_attribute = false,
                 const std::string &prefix = "",
                 const bool &return_mesh = false,
                 dataparser::DataParser::Ptr dataparser_ptr_ = nullptr);
  std::vector<torch::Tensor>
  cull_mesh(dataparser::DataParser::Ptr dataparser_ptr_, torch::Tensor vertices,
            torch::Tensor faces, torch::Tensor vertice_attrs);
};
} // namespace mesher