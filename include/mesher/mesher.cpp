#include "mesher.h"
#include "cumcubes.hpp"
#include "utils/tqdm.hpp"

using namespace std;

namespace mesher {

#ifdef ENABLE_ROS
void tensor_to_mesh(mesh_msgs::MeshGeometry &mesh,
                    mesh_msgs::MeshVertexColors &mesh_color,
                    const torch::Tensor &vertice, const torch::Tensor &face,
                    const torch::Tensor &color) {
  auto vertex_num = vertice.size(0);
  auto face_num = face.size(0);
  mesh.faces.resize(face_num);
  mesh.vertices.resize(vertex_num);
  mesh_color.vertex_colors.resize(vertex_num);

  auto vertice_float64 = vertice.to(torch::kFloat64).contiguous().cpu();
  torch::Tensor color_cat;
  TORCH_CHECK(color.size(-1) == 3)
  color_cat =
      torch::cat({color, torch::ones({color.size(0), 1}, color.options())}, 1)
          .contiguous()
          .cpu();

  memcpy(mesh.vertices.data(), vertice_float64.data_ptr(),
         vertice_float64.numel() * sizeof(double));
  memcpy(mesh_color.vertex_colors.data(), color_cat.data_ptr(),
         color_cat.numel() * sizeof(float));

  auto face_int32 = face.to(torch::kInt32).contiguous().cpu();
  memcpy(mesh.faces.data(), face_int32.data_ptr(),
         face_int32.numel() * sizeof(uint32_t));
}
#endif

Mesher::Mesher() {};

void Mesher::save_mesh(const std::string &output_path, bool vis_attribute,
                       const std::string &prefix, const bool &return_mesh,
                       dataparser::DataParser::Ptr dataparser_ptr_) {
  if (vec_face_.empty()) {
    cout << "\n"
         << "No mesh to save! Please first meshing using 'v' first!" << "\n";
    return;
  }
  cout << "\nSaving mesh to: " << output_path << "\n";

  auto vertice_attrs = torch::cat({vec_face_attr_}, 0).contiguous();
  faces_ = torch::cat({vec_face_}, 0).contiguous();
  vertices_ = torch::cat({vec_vertice_}, 0).contiguous();
  std::string filename = output_path + "/mesh_" + prefix + ".ply";
  if (vertice_attrs.numel() == 0) {
    vertice_attrs = torch::full_like(vertices_, 127).to(torch::kUInt8);
  }
  mc::save_mesh_as_ply(filename, vertices_, faces_, vertice_attrs);
  if (dataparser_ptr_) {
    std::string cull_filename = output_path + "/mesh_culled_" + prefix + ".ply";
    auto cull_results =
        cull_mesh(dataparser_ptr_, vertices_, faces_, vertice_attrs);
    mc::save_mesh_as_ply(cull_filename, cull_results[0], cull_results[1],
                         cull_results[2]);
  }
  vec_face_attr_.clear();
  vec_face_.clear();
  vec_vertice_.clear();
  if (!return_mesh) {
    faces_ = torch::Tensor();
    vertices_ = torch::Tensor();
  }
  printf("\n\033[1;32mExport mesh saved to %s\n\033[0m", filename.c_str());
}

std::vector<torch::Tensor>
Mesher::cull_mesh(dataparser::DataParser::Ptr dataparser_ptr_,
                  torch::Tensor vertices, torch::Tensor faces,
                  torch::Tensor vertice_attrs) {
  int n_imgs = dataparser_ptr_->raw_depth_filelists_.size();
  auto whole_mask =
      torch::ones(vertices.size(0), vertices.options()).to(torch::kBool);
  auto device = vertices.device();

  // Pre-compute constants and allocate reusable tensors
  static const auto K = torch::tensor({{dataparser_ptr_->sensor_.camera.fx, 0.f,
                                        dataparser_ptr_->sensor_.camera.cx},
                                       {0.f, dataparser_ptr_->sensor_.camera.fy,
                                        dataparser_ptr_->sensor_.camera.cy},
                                       {0.f, 0.f, 1.f}},
                                      device);
  static const auto W = dataparser_ptr_->sensor_.camera.width;
  static const auto H = dataparser_ptr_->sensor_.camera.height;

  // Pre-compute homogeneous vertices (only done once)
  const auto ones =
      torch::ones({vertices.size(0), 1}, torch::kFloat32).to(device);
  const auto homo_points = torch::cat({vertices, ones}, 1).reshape({-1, 4, 1});

  // Buffers for reuse in the loop
  auto cam_cord_homo =
      torch::empty({vertices.size(0), 4, 1}, vertices.options());
  auto cam_cord = torch::empty({vertices.size(0), 3, 1}, vertices.options());
  auto uv = torch::empty({vertices.size(0), 2}, vertices.options());
  auto grid = torch::empty({vertices.size(0), 2}, vertices.options());
  auto mask =
      torch::empty(vertices.size(0), vertices.options().dtype(torch::kBool));

  auto iter_bar = tq::trange(n_imgs);
  iter_bar.set_prefix("Culling mesh");

  for (const auto &i : iter_bar) {
    auto pose =
        dataparser_ptr_->get_pose(i, dataparser::DataType::RawDepth).to(device);
    auto depth = dataparser_ptr_->get_depth_image(i).to(device);
    auto w2c = torch::inverse(pose);

    // Transform points to camera space
    cam_cord_homo = w2c.matmul(homo_points);
    cam_cord = cam_cord_homo.slice(1, 0, 3);
    auto z = cam_cord.slice(1, -1);
    auto z_squeezed = z.squeeze();

    // Project to image space
    uv = cam_cord / z.abs();
    uv = K.matmul(uv);
    uv = uv.slice(1, 0, 2).squeeze(-1);

    // Create grid for sampling
    auto grid_x = uv.slice(1, 0, 1) / W;
    auto grid_y = uv.slice(1, 1) / H;
    grid = torch::cat({grid_x, grid_y}, 1);
    grid = 2 * grid - 1; // Convert to [-1,1] range

    // Sample depth from image
    auto input = depth.unsqueeze(0).unsqueeze(1).squeeze(-1);
    auto flow_field = grid.unsqueeze(0).unsqueeze(1);
    auto depth_samples = torch::nn::functional::grid_sample(
                             input, flow_field,
                             torch::nn::functional::GridSampleFuncOptions()
                                 .padding_mode(torch::kZeros)
                                 .align_corners(true))
                             .squeeze();

    // Create visibility mask
    mask = (0 <= z_squeezed) & (uv.select(1, 0) < W) & (uv.select(1, 0) > 0) &
           (uv.select(1, 1) < H) & (uv.select(1, 1) > 0) &
           ((depth_samples + 0.02f) > z_squeezed);

    // Update the global mask
    whole_mask &= ~mask;
  }

  // Cull faces based on vertices
  auto face_mask = ~(whole_mask.index({faces}).all(1));
  auto valid_face_idx = face_mask.nonzero().squeeze();
  faces = faces.index({valid_face_idx});

  return {vertices, faces, vertice_attrs};
}
} // namespace mesher