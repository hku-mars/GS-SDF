#pragma once

#include "utils/sensor_utils/cameras.hpp"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace sensor {

inline Cameras camera_from_intrinsics(int width, int height, float fx, float fy,
                                      float cx, float cy) {
  Cameras camera;
  camera.width = width;
  camera.height = height;
  camera.fx = fx;
  camera.fy = fy;
  camera.cx = cx;
  camera.cy = cy;
  camera.scale = 1.0f;
  camera.model = 0;
  return camera;
}

// render_camera_spec_v1
//   Comments (#) and blank lines are ignored.
//   Each data line: width height fx fy cx cy
//   A single line is broadcast to every pose frame.
inline std::vector<Cameras> load_render_camera_spec(const std::string &spec_path,
                                                   int num_frames) {
  if (spec_path.empty()) {
    return {};
  }
  if (!std::filesystem::exists(spec_path)) {
    throw std::runtime_error("Camera spec file does not exist: " + spec_path);
  }
  if (num_frames <= 0) {
    throw std::runtime_error("Cannot load camera spec for empty pose sequence");
  }

  std::ifstream file(spec_path);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open camera spec file: " + spec_path);
  }

  std::vector<Cameras> cameras;
  std::string line;
  while (std::getline(file, line)) {
    if (line.empty()) {
      continue;
    }
    const auto hash_pos = line.find('#');
    if (hash_pos == 0) {
      continue;
    }
    if (hash_pos != std::string::npos) {
      line = line.substr(0, hash_pos);
    }
    std::istringstream iss(line);
    int width = 0;
    int height = 0;
    float fx = 0.0f;
    float fy = 0.0f;
    float cx = 0.0f;
    float cy = 0.0f;
    if (!(iss >> width >> height >> fx >> fy >> cx >> cy)) {
      continue;
    }
    if (width <= 0 || height <= 0 || fx <= 0.0f || fy <= 0.0f) {
      throw std::runtime_error("Invalid camera intrinsics in spec file: " +
                               spec_path);
    }
    cameras.push_back(camera_from_intrinsics(width, height, fx, fy, cx, cy));
  }

  if (cameras.empty()) {
    throw std::runtime_error("Camera spec file has no valid data lines: " +
                             spec_path);
  }
  if (cameras.size() == 1) {
    cameras.assign(num_frames, cameras.front());
  } else if (static_cast<int>(cameras.size()) != num_frames) {
    throw std::runtime_error(
        "Camera spec frame count (" + std::to_string(cameras.size()) +
        ") does not match pose count (" + std::to_string(num_frames) + ")");
  }

  std::cout << "Loaded render camera spec from: " << spec_path << " ("
            << cameras.size() << " frame(s), " << cameras.front().width << "x"
            << cameras.front().height << ")\n";
  return cameras;
}

inline Cameras resolve_render_camera(const std::vector<Cameras> &spec_cameras,
                                     const Cameras &fallback, int frame_idx) {
  if (spec_cameras.empty()) {
    return fallback;
  }
  if (frame_idx < 0 || frame_idx >= static_cast<int>(spec_cameras.size())) {
    throw std::runtime_error("Camera spec frame index out of range");
  }
  return spec_cameras[frame_idx];
}

} // namespace sensor
