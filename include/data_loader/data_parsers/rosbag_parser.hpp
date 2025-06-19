#pragma once
#include "base_parser.h"
#include "utils/coordinates.h"
#include "utils/sensor_utils/sensors.hpp"
#include <pcl/io/ply_io.h>

#ifdef ENABLE_ROS
#include "cv_bridge/cv_bridge.h"
#include "rosbag/view.h"
#include "sensor_msgs/CompressedImage.h"
#endif

namespace dataparser {
struct Rosbag : DataParser {
  explicit Rosbag(const std::filesystem::path &_bag_path,
                  const torch::Device &_device = torch::kCPU,
                  const bool &_preload = true, const float &_res_scale = 1.0,
                  const int &_dataset_system_type = coords::SystemType::OpenCV,
                  const sensor::Sensors &_sensor = sensor::Sensors(),
                  const int &_ds_pt_num = 1e5)
      : DataParser(_bag_path.parent_path(), _device, _preload, _res_scale,
                   _dataset_system_type, _sensor, _ds_pt_num),
        bag_path_(_bag_path) {

    dataset_name_ = bag_path_.filename();
    dataset_name_ = dataset_name_.replace_extension();

    pose_path_ = dataset_path_ / "color_poses.txt";
    depth_pose_path_ = dataset_path_ / "depth_poses.txt";
    color_path_ = dataset_path_ / "images";
    depth_path_ = dataset_path_ / "depths";
  }

  std::filesystem::path bag_path_, gt_mesh_path_, depth_pose_path_;
  std::string pose_topic, color_topic, depth_topic;

#ifdef ENABLE_ROS
  void parser_bag_to_file(const std::filesystem::path &bag_path,
                          const std::string &pose_topic,
                          const std::string &color_topic,
                          const std::string &depth_topic) {
    assert(std::filesystem::exists(bag_path));

    std::filesystem::create_directories(color_path_);
    std::filesystem::create_directories(depth_path_);

    rosbag::Bag bag(bag_path);

    rosbag::TopicQuery topics({pose_topic, color_topic, depth_topic});
    rosbag::View view(bag, topics);

    int count = 0;
    int bag_size = view.size();
    nav_msgs::OdometryConstPtr pose_msg_prev_ptr, pose_msg_next_ptr;

    sensor_msgs::ImageConstPtr color_msg_ptr;
    sensor_msgs::CompressedImageConstPtr compressed_color_msg_ptr;
    bool is_compressed_color_msg = false;
    if (color_topic.substr(color_topic.find_last_of('/') + 1) == "compressed") {
      is_compressed_color_msg = true;
    }
    int color_count = 0;
    sensor_msgs::PointCloud2ConstPtr depth_msg_ptr;
    int depth_count = 0;

    std::ofstream color_pose_file(pose_path_);
    std::ofstream depth_pose_file(depth_pose_path_);
    for (const rosbag::MessageInstance &m : view) {
      count++;
      std::string topic = m.getTopic();
      std::cout << "\rRead bag message:" << count << "/" << bag_size << ","
                << topic;

      if (topic == pose_topic) {
        // two pose: one for lidar one for camera
        pose_msg_prev_ptr = pose_msg_next_ptr;
        pose_msg_next_ptr = m.instantiate<nav_msgs::Odometry>();

        if (!pose_msg_prev_ptr) {
          continue;
        }

        if (color_msg_ptr || compressed_color_msg_ptr) {
          std_msgs::Header color_msg_header =
              is_compressed_color_msg ? compressed_color_msg_ptr->header
                                      : color_msg_ptr->header;
          auto delta_pose_prev = abs(
              (pose_msg_prev_ptr->header.stamp - color_msg_ptr->header.stamp)
                  .toSec());

          auto delta_pose_next = abs(
              (pose_msg_next_ptr->header.stamp - color_msg_ptr->header.stamp)
                  .toSec());

          auto min_delta = std::min(delta_pose_prev, delta_pose_next);
          if (min_delta < 0.01) {

            std::string filename =
                color_path_ / (std::to_string(count) + ".png");
            cv_bridge::CvImagePtr cv_ptr;
            if (is_compressed_color_msg)
              cv_ptr = cv_bridge::toCvCopy(compressed_color_msg_ptr, "bgr8");
            else
              cv_ptr = cv_bridge::toCvCopy(color_msg_ptr, "bgr8");
            cv::Mat undistorted_img = sensor_.camera.undistort(cv_ptr->image);
            if (cv::imwrite(filename, undistorted_img)) {
              auto nearest_pose_msg = delta_pose_prev < delta_pose_next
                                          ? pose_msg_prev_ptr
                                          : pose_msg_next_ptr;

              auto pos_W_B =
                  torch::tensor({{nearest_pose_msg->pose.pose.position.x},
                                 {nearest_pose_msg->pose.pose.position.y},
                                 { nearest_pose_msg->pose.pose.position.z }},
                                torch::kFloat);
              auto quat_W_B =
                  torch::tensor({nearest_pose_msg->pose.pose.orientation.w,
                                 nearest_pose_msg->pose.pose.orientation.x,
                                 nearest_pose_msg->pose.pose.orientation.y,
                                 nearest_pose_msg->pose.pose.orientation.z},
                                torch::kFloat);
              auto rot_W_B = utils::quat_to_rot(quat_W_B);

              torch::Tensor T_W_B = torch::eye(4, 4);
              T_W_B.index_put_(
                  {torch::indexing::Slice(0, 3), torch::indexing::Slice(0, 3)},
                  rot_W_B);
              T_W_B.index_put_(
                  {torch::indexing::Slice(0, 3), torch::indexing::Slice(3, 4)},
                  pos_W_B);

              auto T_W_C = T_W_B.matmul(sensor_.T_B_C);

              for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                  color_pose_file << T_W_C[i][j].item<float>() << " ";
                }
                color_pose_file << "\n";
              }
              color_count++;
            }

            color_msg_ptr = nullptr;
            compressed_color_msg_ptr = nullptr;
          }
        }

        if (depth_msg_ptr) {
          auto delta_pose_prev = abs(
              (pose_msg_prev_ptr->header.stamp - depth_msg_ptr->header.stamp)
                  .toSec());

          auto delta_pose_next = abs(
              (pose_msg_next_ptr->header.stamp - depth_msg_ptr->header.stamp)
                  .toSec());

          auto min_delta = std::min(delta_pose_prev, delta_pose_next);
          if (min_delta < 0.01) {
            std::string filename =
                depth_path_ / (std::to_string(count) + ".ply");
            pcl::PCLPointCloud2 pcl_pc2;
            pcl_conversions::toPCL(*depth_msg_ptr, pcl_pc2);
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(
                new pcl::PointCloud<pcl::PointXYZ>);
            pcl::fromPCLPointCloud2(pcl_pc2, *cloud);
            if (pcl::io::savePLYFile(filename, *cloud) != -1) {

              auto nearest_pose_msg = delta_pose_prev < delta_pose_next
                                          ? pose_msg_prev_ptr
                                          : pose_msg_next_ptr;

              auto pos_W_B =
                  torch::tensor({{nearest_pose_msg->pose.pose.position.x},
                                 {nearest_pose_msg->pose.pose.position.y},
                                 { nearest_pose_msg->pose.pose.position.z }},
                                torch::kFloat);
              auto quat_W_B =
                  torch::tensor({nearest_pose_msg->pose.pose.orientation.w,
                                 nearest_pose_msg->pose.pose.orientation.x,
                                 nearest_pose_msg->pose.pose.orientation.y,
                                 nearest_pose_msg->pose.pose.orientation.z},
                                torch::kFloat);
              auto rot_W_B = utils::quat_to_rot(quat_W_B);

              torch::Tensor T_W_B = torch::eye(4, 4);
              T_W_B.index_put_(
                  {torch::indexing::Slice(0, 3), torch::indexing::Slice(0, 3)},
                  rot_W_B);
              T_W_B.index_put_(
                  {torch::indexing::Slice(0, 3), torch::indexing::Slice(3, 4)},
                  pos_W_B);

              auto T_W_L = T_W_B.matmul(sensor_.T_B_L);

              for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                  depth_pose_file << T_W_L[i][j].item<float>() << " ";
                }
                depth_pose_file << "\n";
              }
              depth_count++;
            }
            depth_msg_ptr = nullptr;
          }
        }

      } else if (topic == color_topic) {
        if (is_compressed_color_msg)
          compressed_color_msg_ptr =
              m.instantiate<sensor_msgs::CompressedImage>();
        else
          color_msg_ptr = m.instantiate<sensor_msgs::Image>();

      } else if (topic == depth_topic) {
        depth_msg_ptr = m.instantiate<sensor_msgs::PointCloud2>();
      }
    }
  }
#endif

  void load_data() override {
    if (!std::filesystem::exists(pose_path_) ||
        !std::filesystem::exists(depth_pose_path_) ||
        !std::filesystem::exists(color_path_) ||
        !std::filesystem::exists(depth_path_)) {
#ifdef ENABLE_ROS
      parser_bag_to_file(bag_path_, pose_topic, color_topic, depth_topic);
#else
      throw std::runtime_error("No pose or image data found.");
#endif
    }

    color_poses_ = std::get<0>(load_poses(pose_path_, false, 0));
    TORCH_CHECK(color_poses_.size(0) > 0);
    depth_poses_ = std::get<0>(load_poses(depth_pose_path_, false, 0));
    TORCH_CHECK(depth_poses_.size(0) > 0);

    load_colors(".png", "", false, true);
    TORCH_CHECK(color_poses_.size(0) == raw_color_filelists_.size());
    load_depths(".ply", "", false, true);
    TORCH_CHECK(depth_poses_.size(0) == raw_depth_filelists_.size());
  }

  std::vector<at::Tensor> get_distance_ndir_zdirn(const int &idx) override {
    /**
     * @description:
     * @return {distance, ndir, dir_norm}, where ndir.norm = 1;
               {[height width 1], [height width 3], [height width 1]}
     */

    auto pointcloud = get_depth_image(idx);
    // [height width 1]
    auto distance = pointcloud.norm(2, -1, true);
    auto ndir = pointcloud / distance;
    return {distance, ndir, distance};
  }
};
} // namespace dataparser