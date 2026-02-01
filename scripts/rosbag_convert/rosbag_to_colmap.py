#!/usr/bin/env python3
"""
ROS Bag to COLMAP Format Converter

Converts ROS bag data (images + poses + point clouds) to COLMAP txt format
for use with GS-SDF and other 3D reconstruction pipelines.

Usage:
    python rosbag_to_colmap.py \
        --bag_path /path/to/bag.bag \
        --image_topic /camera/image_raw \
        --image_pose_topic /camera/pose \
        --point_topic /lidar/points \
        --point_pose_topic /lidar/pose \
        --output_dir /path/to/output \
        --fx 500 --fy 500 --cx 320 --cy 240 \
        --width 640 --height 480 \
        --k1 0.0 --k2 0.0 --p1 0.0 --p2 0.0

Output structure:
    output_dir/
    ├── images/
    │   ├── 00000.png
    │   ├── 00001.png
    │   └── ...
    ├── depths/
    │   ├── 00000.ply
    │   ├── 00001.ply
    │   └── ...
    └── sparse/
        └── 0/
            ├── cameras.txt
            ├── images.txt
            ├── depths.txt
            └── points3D.ply  (integrated point cloud in world coordinates)
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    import rosbag
    from cv_bridge import CvBridge
    from sensor_msgs.msg import Image, CompressedImage, PointCloud2
    from nav_msgs.msg import Odometry
    import sensor_msgs.point_cloud2 as pc2
except ImportError:
    print("Error: ROS packages not found. Please source your ROS workspace.")
    print("  source /opt/ros/noetic/setup.bash")
    sys.exit(1)

try:
    import open3d as o3d
except ImportError:
    print("Note: open3d not found. Using fallback ASCII PLY writer for point clouds.")
    o3d = None

# Optional: use tqdm for progress bars if available, otherwise fallback to plain iterator
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


def quaternion_from_matrix(matrix: np.ndarray) -> np.ndarray:
    """
    Convert a 3x3 rotation matrix to quaternion (w, x, y, z).

    Args:
        matrix: 3x3 rotation matrix

    Returns:
        Quaternion as [w, x, y, z]
    """
    # Ensure the matrix is 3x3
    R = matrix[:3, :3]

    # Compute quaternion using Shepperd's method
    trace = np.trace(R)

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    return np.array([w, x, y, z])


def odometry_to_pose_matrix(odom: Odometry) -> np.ndarray:
    """
    Convert nav_msgs/Odometry to 4x4 pose matrix (camera-to-world).

    Args:
        odom: Odometry message

    Returns:
        4x4 transformation matrix T_W_C (world to camera frame pose)
    """
    # Extract position
    pos = odom.pose.pose.position
    t = np.array([pos.x, pos.y, pos.z])

    # Extract orientation (quaternion xyzw in ROS)
    ori = odom.pose.pose.orientation
    qx, qy, qz, qw = ori.x, ori.y, ori.z, ori.w

    # Convert quaternion to rotation matrix
    # Using the formula for quaternion to rotation matrix
    R = np.array(
        [
            [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)],
        ]
    )

    # Construct 4x4 matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    return T


def c2w_to_w2c(T_c2w: np.ndarray) -> tuple:
    """
    Convert camera-to-world pose to world-to-camera pose for COLMAP.

    COLMAP uses world-to-camera convention:
    - R_w2c = R_c2w.T
    - t_w2c = -R_w2c @ t_c2w

    Args:
        T_c2w: 4x4 camera-to-world transformation matrix

    Returns:
        Tuple of (quaternion_wxyz, translation) in world-to-camera frame
    """
    R_c2w = T_c2w[:3, :3]
    t_c2w = T_c2w[:3, 3]

    # Convert to world-to-camera
    R_w2c = R_c2w.T
    t_w2c = -R_w2c @ t_c2w

    # Convert rotation to quaternion (w, x, y, z)
    quat = quaternion_from_matrix(R_w2c)

    return quat, t_w2c


def find_closest_pose(
    target_time: float, pose_msgs: list, threshold: float = 0.01
) -> Odometry:
    """
    Find the pose message with timestamp closest to target_time.

    Args:
        target_time: Target timestamp in seconds
        pose_msgs: List of (timestamp, Odometry) tuples
        threshold: Maximum allowed time difference in seconds (default 10ms)

    Returns:
        Closest Odometry message, or None if no match within threshold
    """
    if not pose_msgs:
        return None

    closest_pose = None
    min_delta = float("inf")

    for ts, pose in pose_msgs:
        delta = abs(target_time - ts)
        if delta < min_delta:
            min_delta = delta
            closest_pose = pose

    return closest_pose if min_delta < threshold else None


def undistort_image(
    image: np.ndarray,
    K: np.ndarray,
    dist_coeffs: np.ndarray,
    distortion_model: str = "pinhole",
) -> np.ndarray:
    """
    Undistort an image using camera intrinsics and distortion coefficients.

    Args:
        image: Input distorted image
        K: 3x3 camera intrinsic matrix
        dist_coeffs: Distortion coefficients [k1, k2, p1, p2] or [k1, k2, k3, k4] for fisheye
        distortion_model: "pinhole" (OpenCV) or "fisheye" (equidistant)

    Returns:
        Undistorted image
    """
    if distortion_model == "fisheye":
        # Use fisheye undistortion
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, dist_coeffs, np.eye(3), K, (image.shape[1], image.shape[0]), cv2.CV_16SC2
        )
        undistorted = cv2.remap(image, map1, map2, cv2.INTER_LINEAR)
    else:
        # Standard pinhole/OpenCV distortion model
        undistorted = cv2.undistort(image, K, dist_coeffs)

    return undistorted


def decode_compressed_image(msg: CompressedImage) -> np.ndarray:
    """
    Decode a compressed image message to numpy array.

    Args:
        msg: CompressedImage message

    Returns:
        Decoded image as numpy array (BGR), or None if decoding fails
    """
    np_arr = np.frombuffer(msg.data, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image is None:
        print(f"\nWarning: Failed to decode compressed image (format: {msg.format})")
    return image


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR) between two images.

    Higher PSNR means images are more similar.

    Args:
        img1: First image (numpy array)
        img2: Second image (numpy array)

    Returns:
        PSNR value in dB. Returns infinity if images are identical.
    """
    if img1.shape != img2.shape:
        return 0.0  # Different shapes, consider as different images

    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")  # Identical images

    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


def compute_blur_score(image: np.ndarray) -> float:
    """
    Compute blur score using Laplacian variance method.

    Higher score means sharper image, lower score means blurrier image.

    Args:
        image: Input image (BGR or grayscale)

    Returns:
        Blur score (variance of Laplacian). Typical values:
        - < 100: Very blurry
        - 100-500: Moderately blurry
        - > 500: Sharp
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Compute Laplacian and its variance
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()

    return variance


def pointcloud2_to_xyz(msg: PointCloud2) -> np.ndarray:
    """
    Convert PointCloud2 message to Nx3 numpy array of XYZ points.

    Args:
        msg: PointCloud2 message

    Returns:
        Nx3 array of XYZ coordinates
    """
    points = []
    for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
        points.append([p[0], p[1], p[2]])
    return np.array(points, dtype=np.float64)


def save_pointcloud_ply(points: np.ndarray, filepath: str):
    """
    Save point cloud as PLY file.

    Args:
        points: Nx3 array of XYZ coordinates
        filepath: Output file path
    """
    if o3d is not None:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        o3d.io.write_point_cloud(filepath, pcd)
    else:
        # Fallback: write PLY manually
        with open(filepath, "w") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("end_header\n")
            for p in points:
                f.write(f"{p[0]} {p[1]} {p[2]}\n")


def write_cameras_txt(
    filepath: str, width: int, height: int, fx: float, fy: float, cx: float, cy: float
):
    """
    Write cameras.txt in COLMAP format.

    After undistortion, we use PINHOLE model (no distortion parameters needed).

    Args:
        filepath: Output file path
        width, height: Image dimensions
        fx, fy, cx, cy: Camera intrinsic parameters
    """
    with open(filepath, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("# Number of cameras: 1\n")
        # PINHOLE model: fx, fy, cx, cy
        f.write(f"1 PINHOLE {width} {height} {fx} {fy} {cx} {cy}\n")


def write_images_txt(filepath: str, image_data: list):
    """
    Write images.txt in COLMAP format.

    Format:
        IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
        (empty line for POINTS2D)

    Args:
        filepath: Output file path
        image_data: List of (image_id, quat_wxyz, trans, filename) tuples
    """
    with open(filepath, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(image_data)}\n")

        for img_id, quat, trans, filename in image_data:
            qw, qx, qy, qz = quat
            tx, ty, tz = trans
            f.write(f"{img_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} 1 {filename}\n")
            f.write("\n")  # Empty line for POINTS2D


def write_depths_txt(filepath: str, depth_data: list):
    """
    Write depths.txt in COLMAP-like format for point cloud poses.

    Same format as images.txt but for depth/point cloud data.

    Args:
        filepath: Output file path
        depth_data: List of (depth_id, quat_wxyz, trans, filename) tuples
    """
    with open(filepath, "w") as f:
        f.write("# Depth list with two lines of data per depth:\n")
        f.write("#   DEPTH_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   (empty line)\n")
        f.write(f"# Number of depths: {len(depth_data)}\n")

        for depth_id, quat, trans, filename in depth_data:
            qw, qx, qy, qz = quat
            tx, ty, tz = trans
            f.write(f"{depth_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} 1 {filename}\n")
            f.write("\n")


def parse_rosbag(args):
    """
    Main function to parse ROS bag and export to COLMAP format.

    Args:
        args: Parsed command line arguments
    """
    # Create output directories
    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"
    depths_dir = output_dir / "depths"
    sparse_dir = output_dir / "sparse" / "0"

    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(exist_ok=True)
    depths_dir.mkdir(exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    # Camera intrinsic matrix (distorted)
    K = np.array(
        [[args.fx, 0, args.cx], [0, args.fy, args.cy], [0, 0, 1]], dtype=np.float64
    )

    # Distortion coefficients
    if args.distortion_model == "fisheye":
        dist_coeffs = np.array(
            [args.k1, args.k2, args.k3, args.k4], dtype=np.float64
        )
    else:
        # Pinhole/radial-tangential with optional p3 term
        dist_coeffs = np.array(
            [args.k1, args.k2, args.p1, args.p2, args.p3], dtype=np.float64
        )

    # Prepare undistortion maps and rectified intrinsics
    need_undistort = not args.no_undistort and np.any(dist_coeffs != 0)
    rectified_K = K.copy()
    map1 = None
    map2 = None
    image_size = (args.width, args.height)

    if need_undistort:
        if args.distortion_model == "fisheye":
            # Fisheye model: estimate rectified intrinsics then build maps
            rectified_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                K,
                dist_coeffs,
                image_size,
                np.eye(3),
                balance=0,
                new_size=image_size,
                fov_scale=1.0,
            )
            # Force principal point to the optical center after rectification
            rectified_K[0, 2] = (image_size[0] - 1) * 0.5
            rectified_K[1, 2] = (image_size[1] - 1) * 0.5
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                K, dist_coeffs, np.eye(3), rectified_K, image_size, cv2.CV_16SC2
            )
        else:
            # Pinhole/radial-tangential: compute optimal new K and maps
            print(K)
            print(dist_coeffs)
            print(image_size)
            rectified_K, _ = cv2.getOptimalNewCameraMatrix(
                K, dist_coeffs, image_size, 0, image_size, centerPrincipalPoint=True
            )
            map1, map2 = cv2.initUndistortRectifyMap(
                K, dist_coeffs, np.eye(3), rectified_K, image_size, cv2.CV_16SC2
            )
        print("origin K:\n", K)
        print("undistort K:\n", rectified_K)
        # Downstream we treat the rectified camera as an ideal pinhole
        dist_coeffs = np.zeros_like(dist_coeffs)
    else:
        print("Undistortion skipped (no distortion or disabled)")

    cam_fx, cam_fy = rectified_K[0, 0], rectified_K[1, 1]
    cam_cx, cam_cy = rectified_K[0, 2], rectified_K[1, 2]
    print("Undistortion needed:", need_undistort)

    # Open bag file
    print(f"Opening bag file: {args.bag_path}")
    bag = rosbag.Bag(args.bag_path, "r")

    # Detect compressed image topic from bag info
    bag_info = bag.get_type_and_topic_info()
    is_compressed_image = False
    if args.image_topic in bag_info.topics:
        msg_type = bag_info.topics[args.image_topic].msg_type
        is_compressed_image = "CompressedImage" in msg_type
        print(f"Image topic type: {msg_type} (compressed: {is_compressed_image})")

    # Collect topics to read
    topics = [args.image_pose_topic, args.image_topic]
    if args.point_topic and args.point_pose_topic:
        topics.extend([args.point_pose_topic, args.point_topic])

    # First pass: collect all messages
    print("First pass: collecting messages...")
    image_pose_msgs = []
    image_msgs = []
    point_pose_msgs = []
    point_msgs = []

    bridge = CvBridge()

    total_msgs = sum(
        bag_info.topics[t].message_count for t in topics if t in bag_info.topics
    )

    count = 0
    for topic, msg, t in bag.read_messages(topics=topics):
        count += 1
        timestamp = t.to_sec()

        print(f"\rReading message {count}/{total_msgs}: {topic}", end="")

        if topic == args.image_pose_topic:
            image_pose_msgs.append((timestamp, msg))
        elif topic == args.image_topic:
            image_msgs.append((timestamp, msg))
        elif topic == args.point_pose_topic:
            point_pose_msgs.append((timestamp, msg))
        elif topic == args.point_topic:
            point_msgs.append((timestamp, msg))

    print(
        f"\nCollected: {len(image_pose_msgs)} image poses, {len(image_msgs)} images, "
        f"{len(point_pose_msgs)} point poses, {len(point_msgs)} point clouds"
    )

    # If point_pose_topic is same as image_pose_topic or empty, reuse image poses
    if args.point_pose_topic == args.image_pose_topic or not args.point_pose_topic:
        point_pose_msgs = image_pose_msgs

    if args.skip_point:
        if not args.point_topic:
            print("Error: --skip_point requires --point_topic")
            sys.exit(1)
        if not point_msgs:
            print("Error: --skip_point specified but no point messages found.")
            sys.exit(1)
        if len(point_msgs) != len(image_msgs):
            print(
                f"Error: skip_point requires matched frame counts (images={len(image_msgs)}, points={len(point_msgs)})"
            )
            sys.exit(1)

        max_delta = 0.0
        for idx, ((img_ts, _), (pc_ts, _)) in enumerate(zip(image_msgs, point_msgs)):
            delta = abs(img_ts - pc_ts)
            max_delta = max(max_delta, delta)
            if delta > args.time_threshold:
                print(
                    f"Error: frame timestamp mismatch at index {idx}: |image-point|={delta:.6f}s exceeds threshold {args.time_threshold}ns"
                )
                # sys.exit(1)

        print(
            f"skip_point enabled: frames aligned (count={len(image_msgs)}, max time offset={max_delta:.6f}s)"
        )

    # Helper function to decode and undistort image at given index
    def get_processed_image(idx):
        """Decode and undistort image at given index. Returns (image, pose, reason).

        reason is None if successful, otherwise one of:
          - 'no_pose' : no matching pose found within threshold
          - 'decode'  : image failed to decode or index out of range
        """
        if idx < 0 or idx >= len(image_msgs):
            return None, None, 'decode'
        ts, img_msg = image_msgs[idx]
        closest_pose = find_closest_pose(ts, image_pose_msgs, args.time_threshold)
        
        if closest_pose is None:
            return None, None, 'no_pose'

        if is_compressed_image:
            image = decode_compressed_image(img_msg)
        else:
            image = bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")

        # Check if image decoding failed
        if image is None:
            return None, None, 'decode'

        if need_undistort and map1 is not None and map2 is not None:
            image = cv2.remap(image, map1, map2, cv2.INTER_LINEAR)

        return image, closest_pose, None

    # Second pass: match images with poses and save
    # Strategy:
    # 1. Use PSNR to find candidate images (images different enough from previous export)
    # 2. If candidate is blurry, search nearby images for a sharper replacement
    # 3. Replacement must still satisfy PSNR constraint relative to previous export
    print("Second pass: matching images with poses...")
    image_data = []
    image_count = 0
    skipped_by_psnr = 0
    replaced_by_blur = 0
    prev_exported_image = None  # Store previous exported image for PSNR comparison
    last_processed_idx = -1  # Track the last processed index to avoid going backward
    exported_indices = []  # Track image indices that were actually exported

    # Use tqdm to show image scanning progress
    pbar = tqdm(total=len(image_msgs), desc='Images', unit='img')
    skipped_no_pose = 0
    skipped_decode = 0
    i = 0
    while i < len(image_msgs):
        image, closest_pose, skip_reason = get_processed_image(i)

        if image is None:
            i += 1
            if skip_reason == 'no_pose':
                skipped_no_pose += 1
            elif skip_reason == 'decode':
                skipped_decode += 1
            pbar.update(1)
            pbar.set_postfix(saved=image_count, skipped_psnr=skipped_by_psnr, replaced_blur=replaced_by_blur, skipped_no_pose=skipped_no_pose, skipped_decode=skipped_decode)
            continue

        # PSNR-based filtering: skip if too similar to previous exported image
        if args.psnr_threshold > 0 and prev_exported_image is not None:
            psnr = compute_psnr(image, prev_exported_image)
            if psnr > args.psnr_threshold:
                skipped_by_psnr += 1
                i += 1
                pbar.update(1)
                pbar.set_postfix(saved=image_count, skipped_psnr=skipped_by_psnr, replaced_blur=replaced_by_blur)
                continue

        # This image passes PSNR check - it's a candidate
        candidate_idx = i
        candidate_image = image
        candidate_pose = closest_pose
        candidate_blur = (
            compute_blur_score(image) if args.blur_threshold > 0 else float("inf")
        )

        # If blur check is enabled and candidate is too blurry, search for nearby replacement
        if args.blur_threshold > 0 and candidate_blur < args.blur_threshold:
            # Search nearby images for a sharper one that still satisfies PSNR
            search_radius = args.blur_search_radius
            min_distance = args.min_frame_distance
            best_blur = candidate_blur
            best_idx = candidate_idx
            best_image = candidate_image
            best_pose = candidate_pose

            # Search forward and backward within radius
            for offset in range(1, search_radius + 1):
                for direction in [1, -1]:  # forward and backward
                    search_idx = candidate_idx + offset * direction

                    # Skip if out of bounds or before last processed
                    if search_idx < 0 or search_idx >= len(image_msgs):
                        continue
                    if search_idx <= last_processed_idx:
                        continue

                    # Skip if too close to last exported image (ensure visual parallax)
                    if search_idx - last_processed_idx < min_distance:
                        continue

                    search_image, search_pose, _reason = get_processed_image(search_idx)
                    if search_image is None:
                        continue

                    # Check PSNR constraint relative to previous exported image
                    if prev_exported_image is not None:
                        search_psnr = compute_psnr(search_image, prev_exported_image)
                        if search_psnr > args.psnr_threshold:
                            continue  # Too similar, skip

                    # Check blur score
                    search_blur = compute_blur_score(search_image)
                    if search_blur > best_blur:
                        best_blur = search_blur
                        best_idx = search_idx
                        best_image = search_image
                        best_pose = search_pose

            # Use the best found image (even if still below threshold, use the sharpest one available)
            if best_idx != candidate_idx:
                replaced_by_blur += 1
                candidate_idx = best_idx
                candidate_image = best_image
                candidate_pose = best_pose
                candidate_blur = best_blur

        # Save image (zero-padded filename)
        filename = f"{image_count:05d}.png"
        cv2.imwrite(str(images_dir / filename), candidate_image)

        # Update previous exported image for next PSNR comparison
        prev_exported_image = candidate_image.copy()
        last_processed_idx = candidate_idx

        # Convert pose to COLMAP format (world-to-camera)
        T_c2w = odometry_to_pose_matrix(candidate_pose)
        quat, trans = c2w_to_w2c(T_c2w)

        image_data.append((image_count + 1, quat, trans, filename))
        image_count += 1
        exported_indices.append(candidate_idx)

        pbar.set_postfix(saved=image_count, skipped_psnr=skipped_by_psnr, replaced_blur=replaced_by_blur)

        # Move to next image after the one we just used and update progress
        new_i = max(i + 1, candidate_idx + 1)
        pbar.update(new_i - i)
        i = new_i

    pbar.close()
    print(
        f"\nSaved {image_count} images (skipped {skipped_by_psnr} similar, replaced {replaced_by_blur} blurry)"
    )

    # Third pass: match point clouds with poses and save
    depth_data = []
    depth_count = 0
    all_world_points = []  # Collect all points in world coordinates for points3D.ply

    if args.skip_point and point_msgs and args.point_topic:
        print("Third pass: exporting point clouds matched to exported images (skip_point enabled)...")

        missing_pc_count = 0
        no_pose_count = 0
        empty_pc_count = 0

        pbar = tqdm(exported_indices, desc='Matched point clouds', unit='pc')
        for idx_position, img_idx in enumerate(pbar):
            if img_idx >= len(point_msgs):
                missing_pc_count += 1
                pbar.set_postfix(saved=depth_count, missing=missing_pc_count, no_pose=no_pose_count, empty=empty_pc_count)
                continue

            ts, pc_msg = point_msgs[img_idx]
            closest_pose = find_closest_pose(ts, point_pose_msgs, args.time_threshold)

            if closest_pose is None:
                no_pose_count += 1
                pbar.set_postfix(saved=depth_count, missing=missing_pc_count, no_pose=no_pose_count, empty=empty_pc_count)
                continue

            points_local = pointcloud2_to_xyz(pc_msg)
            if len(points_local) == 0:
                empty_pc_count += 1
                pbar.set_postfix(saved=depth_count, missing=missing_pc_count, no_pose=no_pose_count, empty=empty_pc_count)
                continue

            filename = f"{depth_count:05d}.ply"
            save_pointcloud_ply(points_local, str(depths_dir / filename))

            T_s2w = odometry_to_pose_matrix(closest_pose)
            R_s2w = T_s2w[:3, :3]
            t_s2w = T_s2w[:3, 3]
            points_world = (R_s2w @ points_local.T).T + t_s2w
            all_world_points.append(points_world)

            quat, trans = c2w_to_w2c(T_s2w)
            depth_data.append((depth_count + 1, quat, trans, filename))
            depth_count += 1

            pbar.set_postfix(saved=depth_count, missing=missing_pc_count, no_pose=no_pose_count, empty=empty_pc_count)

        print(f"\nSaved {depth_count} matched point clouds (skipped: missing={missing_pc_count}, no_pose={no_pose_count}, empty={empty_pc_count})")
        total_exported = len(exported_indices)
        if total_exported:
            pct = depth_count / total_exported * 100.0
            print(f"  - matched: {depth_count}/{total_exported} ({pct:.1f}%)")
        else:
            print("  - matched: 0/0 (0.0%)")
    elif point_msgs and args.point_topic:
        print("Third pass: matching point clouds with poses...")

        no_pose_count = 0
        empty_pc_count = 0

        pbar = tqdm(point_msgs, desc='Point clouds', unit='pc')
        for i, (ts, pc_msg) in enumerate(pbar):
            closest_pose = find_closest_pose(ts, point_pose_msgs, args.time_threshold)

            if closest_pose is None:
                no_pose_count += 1
                pbar.set_postfix(saved=depth_count, no_pose=no_pose_count, empty=empty_pc_count)
                continue

            # Convert to XYZ array (in sensor frame)
            points_local = pointcloud2_to_xyz(pc_msg)

            if len(points_local) == 0:
                empty_pc_count += 1
                pbar.set_postfix(saved=depth_count, no_pose=no_pose_count, empty=empty_pc_count)
                continue

            # Save point cloud (in local/sensor frame, zero-padded filename)
            filename = f"{depth_count:05d}.ply"
            save_pointcloud_ply(points_local, str(depths_dir / filename))

            # Get pose (sensor-to-world transformation)
            T_s2w = odometry_to_pose_matrix(closest_pose)

            # Transform points to world coordinates for integrated point cloud
            R_s2w = T_s2w[:3, :3]
            t_s2w = T_s2w[:3, 3]
            points_world = (R_s2w @ points_local.T).T + t_s2w
            all_world_points.append(points_world)

            # Convert pose to COLMAP format (world-to-sensor)
            quat, trans = c2w_to_w2c(T_s2w)

            depth_data.append((depth_count + 1, quat, trans, filename))
            depth_count += 1

            pbar.set_postfix(saved=depth_count, no_pose=no_pose_count, empty=empty_pc_count)
        print(f"\nSaved {depth_count} point clouds (skipped: no_pose={no_pose_count}, empty={empty_pc_count})")
        total_point_msgs = len(point_msgs)
        if total_point_msgs:
            pct = depth_count / total_point_msgs * 100.0
            print(f"  - matched: {depth_count}/{total_point_msgs} ({pct:.1f}%)")
        else:
            print("  - matched: 0/0 (0.0%)")

    bag.close()

    # Write COLMAP format files to sparse/0/
    print("Writing COLMAP format files...")

    write_cameras_txt(
        str(sparse_dir / "cameras.txt"),
        args.width,
        args.height,
        cam_fx,
        cam_fy,
        cam_cx,
        cam_cy,
    )

    write_images_txt(str(sparse_dir / "images.txt"), image_data)

    write_depths_txt(str(sparse_dir / "depths.txt"), depth_data)

    # Create integrated point cloud in world coordinates
    if all_world_points:
        print("Creating integrated point cloud (points3D.ply)...")
        integrated_points = np.vstack(all_world_points)
        print(f"  Total points before downsampling: {len(integrated_points)}")

        # Downsample if too many points (limit to ~2M points to avoid OOM)
        max_points = 2_000_000
        if len(integrated_points) > max_points:
            step = len(integrated_points) // max_points
            integrated_points = integrated_points[::step]
            print(f"  Downsampled to: {len(integrated_points)} points")

        save_pointcloud_ply(integrated_points, str(sparse_dir / "points3D.ply"))

    print(f"\nDone! Output saved to: {output_dir}")
    print(f"  - sparse/0/cameras.txt: 1 camera")
    print(f"  - sparse/0/images.txt: {len(image_data)} images")
    print(f"  - sparse/0/depths.txt: {len(depth_data)} point clouds")
    if all_world_points:
        print(f"  - sparse/0/points3D.ply: {len(integrated_points)} points")


def main():
    parser = argparse.ArgumentParser(
        description="Convert ROS bag to COLMAP format for GS-SDF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with FAST-LIVO2 bag
  python rosbag_to_colmap.py \\
      --bag_path fast_livo2.bag \\
      --image_topic /origin_img \\
      --image_pose_topic /aft_mapped_to_init \\
      --point_topic /cloud_registered_body \\
      --point_pose_topic /aft_mapped_to_init \\
      --output_dir ./colmap_output \\
      --fx 500 --fy 500 --cx 320 --cy 240 \\
      --width 640 --height 480

  # With distortion correction
  python rosbag_to_colmap.py \\
      --bag_path data.bag \\
      --image_topic /camera/image_raw/compressed \\
      --image_pose_topic /camera/odom \\
      --output_dir ./output \\
      --fx 517.3 --fy 516.5 --cx 318.6 --cy 239.5 \\
      --width 640 --height 480 \\
      --k1 -0.2 --k2 0.1 --p1 0.0 --p2 0.0
        """,
    )

    # Input/output arguments
    parser.add_argument(
        "--bag_path", type=str, required=True, help="Path to ROS bag file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for COLMAP format data",
    )

    # Topic arguments
    parser.add_argument(
        "--image_topic",
        type=str,
        required=True,
        help="ROS topic for images (add /compressed suffix for compressed images)",
    )
    parser.add_argument(
        "--image_pose_topic",
        type=str,
        required=True,
        help="ROS topic for image poses (nav_msgs/Odometry)",
    )
    parser.add_argument(
        "--point_topic",
        type=str,
        default="",
        help="ROS topic for point clouds (optional)",
    )
    parser.add_argument(
        "--point_pose_topic",
        type=str,
        default="",
        help="ROS topic for point cloud poses (optional, defaults to image_pose_topic)",
    )
    parser.add_argument(
        "--skip_point",
        action="store_true",
        help="Skip point cloud when image frame is skipped; requires synchronized image/point frames",
    )

    # Camera intrinsic arguments
    parser.add_argument("--fx", type=float, required=True, help="Focal length x")
    parser.add_argument("--fy", type=float, required=True, help="Focal length y")
    parser.add_argument("--cx", type=float, required=True, help="Principal point x")
    parser.add_argument("--cy", type=float, required=True, help="Principal point y")
    parser.add_argument("--width", type=int, required=True, help="Image width")
    parser.add_argument("--height", type=int, required=True, help="Image height")

    # Distortion arguments
    parser.add_argument(
        "--k1", type=float, default=0.0, help="Distortion coefficient k1"
    )
    parser.add_argument(
        "--k2", type=float, default=0.0, help="Distortion coefficient k2"
    )
    parser.add_argument(
        "--p1", type=float, default=0.0, help="Distortion coefficient p1 (tangential)"
    )
    parser.add_argument(
        "--p2", type=float, default=0.0, help="Distortion coefficient p2 (tangential)"
    )
    parser.add_argument(
        "--k3", type=float, default=0.0, help="Distortion coefficient k3 (for fisheye)"
    )
    parser.add_argument(
        "--k4", type=float, default=0.0, help="Distortion coefficient k4 (for fisheye)"
    )
    parser.add_argument(
        "--p3", type=float, default=0.0, help="Distortion coefficient p3 (optional tangential term)"
    )
    parser.add_argument(
        "--distortion_model",
        type=str,
        default="pinhole",
        choices=["pinhole", "fisheye"],
        help="Distortion model: pinhole (k1,k2,p1,p2) or fisheye (k1,k2,k3,k4)",
    )
    parser.add_argument(
        "--no_undistort",
        action="store_true",
        help="Skip undistortion (use if images are already rectified)",
    )

    # Matching arguments
    parser.add_argument(
        "--time_threshold",
        type=float,
        default=0.01,
        help="Maximum time difference for pose matching in seconds (default: 0.01)",
    )

    # Image filtering arguments
    parser.add_argument(
        "--psnr_threshold",
        type=float,
        default=0.0,
        help="PSNR threshold for filtering similar images. "
        "Images with PSNR > threshold compared to previous saved image will be skipped. "
        "Set to 0 to disable (default: 0). Typical values: 25-35 dB",
    )
    parser.add_argument(
        "--blur_threshold",
        type=float,
        default=0.0,
        help="Blur threshold for filtering blurry images (Laplacian variance). "
        "Images with blur score < threshold will be skipped. "
        "Set to 0 to disable (default: 0). Typical values: 100-500",
    )
    parser.add_argument(
        "--blur_search_radius",
        type=int,
        default=10,
        help="Search radius for finding sharper replacement when candidate is blurry. "
        "Will search +/- this many frames around the blurry candidate. (default: 10)",
    )
    parser.add_argument(
        "--min_frame_distance",
        type=int,
        default=1,
        help="Minimum frame distance from last exported image when searching for "
        "blur replacement. Ensures visual parallax between frames. (default: 1)",
    )

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.bag_path):
        print(f"Error: Bag file not found: {args.bag_path}")
        sys.exit(1)

    parse_rosbag(args)


if __name__ == "__main__":
    main()
