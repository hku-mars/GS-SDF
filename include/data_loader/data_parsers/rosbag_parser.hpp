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

    color_pose_path_ = dataset_path_ / "color_poses.txt";
    depth_pose_path_ = dataset_path_ / "depth_poses.txt";
    color_path_ = dataset_path_ / "images";
    depth_path_ = dataset_path_ / "depths";
  }

  std::filesystem::path bag_path_, gt_mesh_path_, depth_pose_path_;
  std::string color_pose_topic, color_topic, depth_pose_topic, depth_topic;

#ifdef ENABLE_ROS
  void parser_bag_to_file(const std::filesystem::path &bag_path,
                          const std::string &color_pose_topic,
                          const std::string &color_topic,
                          const std::string &depth_pose_topic,
                          const std::string &depth_topic) {
    assert(std::filesystem::exists(bag_path));

    std::filesystem::create_directories(color_path_);
    std::filesystem::create_directories(depth_path_);

    rosbag::Bag bag(bag_path);

    rosbag::TopicQuery topics(
        {color_pose_topic, color_topic, depth_pose_topic, depth_topic});
    rosbag::View view(bag, topics);

    // First pass: collect all messages by type
    std::vector<nav_msgs::OdometryConstPtr> color_pose_msgs, depth_pose_msgs;
    std::vector<sensor_msgs::ImageConstPtr> color_msgs;
    std::vector<sensor_msgs::CompressedImageConstPtr> compressed_color_msgs;
    std::vector<sensor_msgs::PointCloud2ConstPtr> depth_msgs;

    bool is_compressed_color_msg = false;
    if (color_topic.substr(color_topic.find_last_of('/') + 1) == "compressed") {
      is_compressed_color_msg = true;
    }

    int count = 0;
    int bag_size = view.size();

    std::cout << "First pass: collecting all messages..." << std::endl;
    for (const rosbag::MessageInstance &m : view) {
      count++;
      std::string topic = m.getTopic();
      std::cout << "\rRead bag message:" << count << "/" << bag_size << ","
                << topic;

      if (topic == color_pose_topic) {
        color_pose_msgs.emplace_back(m.instantiate<nav_msgs::Odometry>());
      } else if (topic == color_topic) {
        if (is_compressed_color_msg) {
          compressed_color_msgs.emplace_back(
              m.instantiate<sensor_msgs::CompressedImage>());
        } else {
          color_msgs.emplace_back(m.instantiate<sensor_msgs::Image>());
        }
      } else if (topic == depth_pose_topic) {
        depth_pose_msgs.emplace_back(m.instantiate<nav_msgs::Odometry>());
      } else if (topic == depth_topic) {
        depth_msgs.emplace_back(m.instantiate<sensor_msgs::PointCloud2>());
      }
    }

    if ((color_pose_topic == depth_pose_topic) || depth_pose_topic.empty()) {
      depth_pose_msgs = color_pose_msgs;
    }

    std::cout << std::endl
              << "Collected " << color_pose_msgs.size() << " color poses, "
              << (is_compressed_color_msg ? compressed_color_msgs.size()
                                          : color_msgs.size())
              << " color images, " << depth_pose_msgs.size() << " depth poses, "
              << depth_msgs.size() << " depth clouds" << std::endl;

    // Helper function to find closest timestamp match
    auto find_closest_pose =
        [&](const ros::Time &target_time,
            std::vector<nav_msgs::OdometryConstPtr> &pose_msgs)
        -> nav_msgs::OdometryConstPtr {
      if (pose_msgs.empty())
        return nullptr;

      nav_msgs::OdometryConstPtr closest_pose = pose_msgs[0];
      double min_delta =
          abs((target_time - closest_pose->header.stamp).toSec());

      for (const auto &pose : pose_msgs) {
        double delta = abs((target_time - pose->header.stamp).toSec());
        if (delta < min_delta) {
          min_delta = delta;
          closest_pose = pose;
        }
      }

      return (min_delta < 0.01) ? closest_pose : nullptr;
    };

    std::ofstream color_pose_file(color_pose_path_);
    std::ofstream depth_pose_file(depth_pose_path_);

    int color_count = 0;
    int depth_count = 0;

    // Second pass: match color messages with closest poses
    std::cout << "Second pass: matching color images with poses..."
              << std::endl;
    if (is_compressed_color_msg) {
      for (size_t i = 0; i < compressed_color_msgs.size(); i++) {
        const auto &color_msg = compressed_color_msgs[i];
        auto closest_pose =
            find_closest_pose(color_msg->header.stamp, color_pose_msgs);

        if (closest_pose) {
          std::string filename =
              color_path_ / (std::to_string(color_count) + ".png");
          cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(color_msg, "bgr8");
          cv::Mat undistorted_img = sensor_.camera.undistort(cv_ptr->image);

          if (cv::imwrite(filename, undistorted_img)) {
            auto pos_W_B =
                torch::tensor({{closest_pose->pose.pose.position.x},
                               {closest_pose->pose.pose.position.y},
                               { closest_pose->pose.pose.position.z }},
                              torch::kFloat);
            auto quat_W_B =
                torch::tensor({closest_pose->pose.pose.orientation.w,
                               closest_pose->pose.pose.orientation.x,
                               closest_pose->pose.pose.orientation.y,
                               closest_pose->pose.pose.orientation.z},
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
        }
        std::cout << "\rProcessed color image " << (i + 1) << "/"
                  << compressed_color_msgs.size();
      }
    } else {
      for (size_t i = 0; i < color_msgs.size(); i++) {
        const auto &color_msg = color_msgs[i];
        auto closest_pose =
            find_closest_pose(color_msg->header.stamp, color_pose_msgs);

        if (closest_pose) {
          std::string filename =
              color_path_ / (std::to_string(color_count) + ".png");
          cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(color_msg, "bgr8");
          cv::Mat undistorted_img = sensor_.camera.undistort(cv_ptr->image);

          if (cv::imwrite(filename, undistorted_img)) {
            auto pos_W_B =
                torch::tensor({{closest_pose->pose.pose.position.x},
                               {closest_pose->pose.pose.position.y},
                               { closest_pose->pose.pose.position.z }},
                              torch::kFloat);
            auto quat_W_B =
                torch::tensor({closest_pose->pose.pose.orientation.w,
                               closest_pose->pose.pose.orientation.x,
                               closest_pose->pose.pose.orientation.y,
                               closest_pose->pose.pose.orientation.z},
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
        }
        std::cout << "\rProcessed color image " << (i + 1) << "/"
                  << color_msgs.size();
      }
    }

    // Third pass: match depth messages with closest poses
    std::cout << std::endl
              << "Third pass: matching depth clouds with poses..." << std::endl;
    for (size_t i = 0; i < depth_msgs.size(); i++) {
      const auto &depth_msg = depth_msgs[i];
      auto closest_pose =
          find_closest_pose(depth_msg->header.stamp, depth_pose_msgs);

      if (closest_pose) {
        std::string filename =
            depth_path_ / (std::to_string(depth_count) + ".ply");
        pcl::PCLPointCloud2 pcl_pc2;
        pcl_conversions::toPCL(*depth_msg, pcl_pc2);
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(
            new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromPCLPointCloud2(pcl_pc2, *cloud);

        if (pcl::io::savePLYFile(filename, *cloud) != -1) {
          auto pos_W_B = torch::tensor({{closest_pose->pose.pose.position.x},
                                        {closest_pose->pose.pose.position.y},
                                        { closest_pose->pose.pose.position.z }},
                                       torch::kFloat);
          auto quat_W_B = torch::tensor({closest_pose->pose.pose.orientation.w,
                                         closest_pose->pose.pose.orientation.x,
                                         closest_pose->pose.pose.orientation.y,
                                         closest_pose->pose.pose.orientation.z},
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
      }
      std::cout << "\rProcessed depth cloud " << (i + 1) << "/"
                << depth_msgs.size();
    }

    std::cout << std::endl
              << "Processing complete. Color images: " << color_count
              << ", Depth clouds: " << depth_count << std::endl;
  }
#endif

  void load_data() override {
    if (!std::filesystem::exists(color_pose_path_) ||
        !std::filesystem::exists(depth_pose_path_) ||
        !std::filesystem::exists(color_path_) ||
        !std::filesystem::exists(depth_path_)) {
#ifdef ENABLE_ROS
      parser_bag_to_file(bag_path_, color_pose_topic, color_topic,
                         depth_pose_topic, depth_topic);
#else
      throw std::runtime_error("No pose or image data found.");
#endif
    }

    auto color_info = load_poses(color_pose_path_, false, 0);
    color_poses_ = std::get<0>(color_info);
    TORCH_CHECK(color_poses_.size(0) > 0);

    cameras_[0] = sensor_.camera;
    // Disable distortion for ROS data, as the exported images have
    // already been undistorted
    cameras_[0].distortion_ = false;
    color_camera_ids_.resize(color_poses_.size(0), 0);

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