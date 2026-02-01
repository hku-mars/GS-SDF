# ROS Bag to COLMAP Format Converter

A Python script to convert ROS bag data (images + poses + point clouds) to COLMAP txt format for use with GS-SDF and other 3D reconstruction pipelines.

## Features

- Parses ROS bag files with image, pose, and point cloud topics
- Timestamp-based matching of sensor data with poses (configurable threshold)
- Supports both raw and compressed image topics
- Image undistortion with pinhole or fisheye distortion models
- Generates integrated point cloud in world coordinates
- Outputs COLMAP-compatible directory structure

## Requirements

- Python 3.x
- ROS (tested with ROS Noetic)
- OpenCV (`cv2`)
- NumPy
- Open3D (optional, for faster PLY export)

```bash
# Source ROS workspace
source /opt/ros/noetic/setup.bash

# Optional: Install open3d for faster point cloud export
pip install open3d==0.18.0
```

## Output Structure

```
output_dir/
├── images/              # Undistorted images
│   ├── 00000.png
│   ├── 00001.png
│   └── ...
├── depths/              # Individual point clouds (sensor frame)
│   ├── 00000.ply
│   ├── 00001.ply
│   └── ...
└── sparse/
    └── 0/
        ├── cameras.txt   # Camera intrinsics (PINHOLE model)
        ├── images.txt    # Image poses (COLMAP W2C format)
        ├── depths.txt    # Point cloud poses
        └── points3D.ply  # Integrated point cloud (world coordinates)
```

## Usage

### Basic Usage

```bash
python rosbag_to_colmap.py \
    --bag_path /path/to/your.bag \
    --image_topic /camera/image_raw \
    --image_pose_topic /camera/odom \
    --output_dir /path/to/output \
    --fx 500 --fy 500 --cx 320 --cy 240 \
    --width 640 --height 480
```

### With Point Clouds

```bash
python rosbag_to_colmap.py \
    --bag_path /path/to/your.bag \
    --image_topic /camera/image_raw \
    --image_pose_topic /camera/odom \
    --point_topic /lidar/points \
    --point_pose_topic /lidar/odom \
    --output_dir /path/to/output \
    --fx 500 --fy 500 --cx 320 --cy 240 \
    --width 640 --height 480
```

### With Distortion Correction

```bash
python rosbag_to_colmap.py \
    --bag_path /path/to/your.bag \
    --image_topic /camera/image_raw \
    --image_pose_topic /camera/odom \
    --output_dir /path/to/output \
    --fx 863.4241 --fy 863.4171 --cx 640.6808 --cy 518.3392 \
    --width 1280 --height 1024 \
    --k1=-0.1080 --k2=0.1050 --p1=-1.2872e-04 --p2=5.7923e-05
```

### With PSNR Filtering (Skip Similar Images)

```bash
python rosbag_to_colmap.py \
    --bag_path /path/to/your.bag \
    --image_topic /camera/image_raw \
    --image_pose_topic /camera/odom \
    --output_dir /path/to/output \
    --fx 500 --fy 500 --cx 320 --cy 240 \
    --width 640 --height 480 \
    --psnr_threshold 30
```

### With Compressed Images

```bash
python rosbag_to_colmap.py \
    --bag_path /path/to/your.bag \
    --image_topic /camera/image_raw/compressed \
    --image_pose_topic /camera/odom \
    --output_dir /path/to/output \
    --fx 500 --fy 500 --cx 320 --cy 240 \
    --width 640 --height 480
```

### Example: R3Live Dataset

```bash
python rosbag_to_colmap.py \
    --bag_path /media/chrisliu/T9/Datasets/R3Live/r3live_hku_park_00.bag \
    --image_topic /track_img \
    --image_pose_topic /camera_odom \
    --point_topic /cloud_registered_body \
    --point_pose_topic /aft_mapped_to_init \
    --output_dir /media/chrisliu/T9/Datasets/R3Live/r3live_hku_park_00_colmap \
    --fx 863.4241 --fy 863.4171 --cx 640.6808 --cy 518.3392 \
    --width 1280 --height 1024 \
    --k1=-0.1080 --k2=0.1050 --p1=-1.2872e-04 --p2=5.7923e-05 \
    --psnr_threshold 15 \
    --blur_threshold 100
```

### Example: FAST-LIVO2 Dataset

```bash
python rosbag_to_colmap.py \
    --bag_path /path/to/fast_livo2_campus.bag \
    --image_topic /origin_img \
    --image_pose_topic /aft_mapped_to_init \
    --point_topic /cloud_registered_body \
    --point_pose_topic /aft_mapped_to_init \
    --output_dir /path/to/fast_livo2_campus_colmap \
    --fx 863.4241 --fy 863.4171 --cx 640.6808 --cy 518.3392 \
    --width 1280 --height 1024
```

## Arguments

### Required Arguments

| Argument | Description |
|----------|-------------|
| `--bag_path` | Path to ROS bag file |
| `--output_dir` | Output directory for COLMAP format data |
| `--image_topic` | ROS topic for images (add `/compressed` suffix for compressed images) |
| `--image_pose_topic` | ROS topic for image poses (`nav_msgs/Odometry`) |
| `--fx` | Focal length x |
| `--fy` | Focal length y |
| `--cx` | Principal point x |
| `--cy` | Principal point y |
| `--width` | Image width |
| `--height` | Image height |

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--point_topic` | `""` | ROS topic for point clouds (`sensor_msgs/PointCloud2`) |
| `--point_pose_topic` | `""` | ROS topic for point cloud poses (defaults to `image_pose_topic`) |
| `--k1` | `0.0` | Distortion coefficient k1 |
| `--k2` | `0.0` | Distortion coefficient k2 |
| `--p1` | `0.0` | Distortion coefficient p1 (tangential) |
| `--p2` | `0.0` | Distortion coefficient p2 (tangential) |
| `--k3` | `0.0` | Distortion coefficient k3 (for fisheye) |
| `--k4` | `0.0` | Distortion coefficient k4 (for fisheye) |
| `--distortion_model` | `pinhole` | Distortion model: `pinhole` (k1,k2,p1,p2) or `fisheye` (k1,k2,k3,k4) |
| `--no_undistort` | `False` | Skip undistortion (use if images are already rectified) |
| `--time_threshold` | `0.01` | Maximum time difference for pose matching in seconds |
| `--psnr_threshold` | `0.0` | PSNR threshold for filtering similar images. Images with PSNR > threshold compared to previous saved image will be skipped. Set to 0 to disable. Typical values: 25-35 dB |
| `--blur_threshold` | `0.0` | Blur threshold for filtering blurry images (Laplacian variance). Images with blur score < threshold will be skipped. Set to 0 to disable. Typical values: 100-500 |

## Notes

### Negative Values in Arguments

When passing negative values for distortion coefficients, use the `=` syntax to avoid parsing issues:

```bash
# Correct:
--k1=-0.1080 --p1=-1.2872e-04

# May cause issues:
--k1 -0.1080 --p1 -1.2872e-04
```

### Pose Convention

- Input poses are assumed to be in **sensor-to-world** (camera/lidar-to-world) frame
- Output poses in `images.txt` and `depths.txt` are in **COLMAP's world-to-camera** convention
- The `points3D.ply` contains points in **world coordinates**

### Point Cloud Downsampling

The integrated point cloud (`points3D.ply`) is automatically downsampled to ~2 million points to avoid memory issues. Individual frame point clouds in `depths/` are saved at full resolution.

### PSNR-Based Image Filtering

When `--psnr_threshold` is set (e.g., 30), the script compares each new image with the previously saved image using PSNR (Peak Signal-to-Noise Ratio). If the PSNR is higher than the threshold (meaning images are very similar), the current image is skipped.

- **Higher PSNR** = more similar images
- **Typical threshold values**: 25-35 dB
- **Use case**: Reduce redundant frames when the camera is stationary or moving slowly

### Blur Detection

When `--blur_threshold` is set (e.g., 100), the script computes a blur score for each image using the Laplacian variance method. Images with a blur score below the threshold are considered blurry and will be skipped.

- **Higher blur score** = sharper image
- **Use case**: Filter out motion-blurred or out-of-focus frames
- **Note**: Blur detection is applied before PSNR filtering to avoid comparing against blurry images

#### Methodology: Laplacian Variance

The blur detection uses the **Laplacian variance** method, which is a simple and effective approach:

1. **Convert to grayscale**: The image is converted to grayscale
2. **Apply Laplacian operator**: Computes second-order derivatives to detect edges using the kernel:
   ```
   Laplacian Kernel:     [ 0  1  0 ]
                         [ 1 -4  1 ]
                         [ 0  1  0 ]
   
   L(x,y) = I(x-1,y) + I(x+1,y) + I(x,y-1) + I(x,y+1) - 4·I(x,y)
   ```
3. **Compute variance**: The variance of the Laplacian response measures edge strength
   ```
   μ = (1/N) Σ L(x,y)           # Mean of Laplacian values
   
   Var = (1/N) Σ (L(x,y) - μ)²  # Variance
   
   blur_score = Var(L)
   ```
   where N is the total number of pixels.

**Implementation** (OpenCV):
```python
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
laplacian = cv2.Laplacian(gray, cv2.CV_64F)
blur_score = laplacian.var()
```

**Why it works**: Sharp images have strong edges with high-frequency content, resulting in high Laplacian variance. Blurry images have smoothed edges with low-frequency content, resulting in low variance.

#### Blur Score Interpretation

| Blur Score | Image Quality |
|------------|---------------|
| < 50 | Very blurry (motion blur, out of focus) |
| 50-100 | Moderately blurry |
| 100-300 | Acceptable sharpness |
| > 300 | Sharp/crisp image |

#### How to Choose Blur Threshold

1. **Conservative** (`--blur_threshold 50`): Filter only very blurry images, keep most frames
2. **Moderate** (`--blur_threshold 100`): Good balance for most SLAM datasets
3. **Aggressive** (`--blur_threshold 150-200`): Keep only sharp images

#### Factors Affecting Blur Score

- **Image resolution**: Higher resolution → higher scores
- **Scene content**: Textured scenes → higher scores; uniform areas → lower scores
- **Motion speed**: Fast motion → more blur → lower scores

For outdoor SLAM datasets (e.g., R3Live), a threshold of **50-100** is recommended.

### Supported Message Types

- **Images**: `sensor_msgs/Image`, `sensor_msgs/CompressedImage`
- **Poses**: `nav_msgs/Odometry`
- **Point Clouds**: `sensor_msgs/PointCloud2`

## Use with GS-SDF

After conversion, you can use the output with GS-SDF:

```bash
# Run GS-SDF with the converted dataset
rosrun neural_mapping neural_mapping_node train \
    src/GS-SDF/config/colmap/your_config.yaml \
    /path/to/output
```

## Troubleshooting

### "ROS packages not found"

Make sure to source your ROS workspace:

```bash
source /opt/ros/noetic/setup.bash
```

### "open3d not found"

This is just a note, not an error. The script will use a fallback ASCII PLY writer. For faster export, install open3d:

```bash
pip install open3d==0.18.0
```

### Many point clouds dropped

If many point clouds are dropped (not matched with poses), try increasing the time threshold:

```bash
--time_threshold 0.1  # 100ms instead of default 10ms
```

This often happens when image and point cloud pose topics have different publishing rates.
