#include "neural_mapping/neural_mapping.h"
#ifdef ENABLE_ROS
#include <ros/ros.h>
#else
#include <thread>
#endif

#ifndef _WIN32
#define BACKWARD_HAS_DW 1
#include "backward.hpp"
namespace backward {
backward::SignalHandling sh;
}
#endif

/**
 * @brief Main entry point for the neural_mapping node
 *
 * Supports two operating modes:
 * - "view": Visualizes a pretrained model
 * - "train": Trains a new neural mapping model
 */
int main(int argc, char **argv) {
  // Set random seeds for reproducibility
  torch::manual_seed(0);
  torch::cuda::manual_seed_all(0);

#ifdef ENABLE_ROS
  // Initialize ROS node
  ros::init(argc, argv, "neural_mapping");
  ros::NodeHandle nh("neural_mapping");
  ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME,
                                 ros::console::levels::Info);
#endif

  // Check for minimum required arguments
  if (argc < 2) {
    std::cerr << "Usage: neural_mapping_node <mode> [args...]\n"
              << "  Modes: view <pretrained_path> | train <config_path> "
                 "<data_path>\n";
    return 1;
  }

  std::string mode = std::string(argv[1]);
  NeuralSLAM::Ptr neural_mapping_ptr;

  try {
    if (mode == "view") {
      // View mode - load pretrained model
      if (argc != 3) {
        std::cerr << "Usage: neural_mapping_node view <pretrained_path>\n";
        return 1;
      }

      auto pretrained_path = std::filesystem::path(argv[2]);
      auto config_path = pretrained_path / "model/config/scene/config.yaml";

#ifdef ENABLE_ROS
      neural_mapping_ptr = std::make_shared<NeuralSLAM>(nh, 0, config_path);
#else
      neural_mapping_ptr = std::make_shared<NeuralSLAM>(0, config_path);
#endif
      std::cout << "View mode initialized with model: " << pretrained_path
                << std::endl;
    } else if (mode == "train") {
      // Train mode - create new model
      if (argc != 4) {
        std::cerr
            << "Usage: neural_mapping_node train <config_path> <data_path>\n";
        return 1;
      }

      std::string config_path = std::string(argv[2]);
      std::string data_path = std::string(argv[3]);

#ifdef ENABLE_ROS
      neural_mapping_ptr =
          std::make_shared<NeuralSLAM>(nh, 1, config_path, data_path);
#else
      neural_mapping_ptr =
          std::make_shared<NeuralSLAM>(1, config_path, data_path);
#endif
      std::cout << "Training mode initialized with config: " << config_path
                << std::endl;
    } else {
      std::cerr << "Invalid mode: " << mode << "\n";
      return 1;
    }

#ifdef ENABLE_ROS
    ros::spin();
#else
    // Wait indefinitely in non-ROS mode
    std::condition_variable cv;
    std::mutex cv_m;
    std::unique_lock<std::mutex> lock(cv_m);
    cv.wait(lock);
#endif
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << '\n';
    return 1;
  }

  return 0;
}