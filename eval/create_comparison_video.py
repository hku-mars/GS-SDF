#!/usr/bin/env python3
"""
Script to create a side-by-side comparison video from gt and renders images.
Left side: GT images, Right side: Rendered images
"""

import os
import cv2
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm

def get_sorted_image_files(directory):
    """Get sorted list of image files from directory (supports jpg, jpeg, png, bmp, tiff)"""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    image_files = []
    
    for file in os.listdir(directory):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(file)
    
    # Sort files by timestamp (extracted from filename before '+' or '-')
    def extract_timestamp(filename):
        # Handle different filename patterns
        name_part = os.path.splitext(filename)[0]  # Remove extension
        if '+' in name_part:
            return int(name_part.split('+')[0].split('-')[-1])
        elif '-' in name_part:
            # Extract the last numeric part for timestamps
            parts = name_part.split('-')
            for part in reversed(parts):
                if part.isdigit():
                    return int(part)
        # Fallback: try to extract any numbers from filename
        import re
        numbers = re.findall(r'\d+', name_part)
        return int(numbers[0]) if numbers else 0
    
    image_files.sort(key=extract_timestamp)
    return image_files

def resize_image_to_match(img1, img2):
    """Resize images to have the same height and width"""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Use the minimum height and proportional width
    target_height = min(h1, h2)
    
    # Calculate new widths maintaining aspect ratio
    new_w1 = int(w1 * target_height / h1)
    new_w2 = int(w2 * target_height / h2)
    
    img1_resized = cv2.resize(img1, (new_w1, target_height))
    img2_resized = cv2.resize(img2, (new_w2, target_height))
    
    return img1_resized, img2_resized

def create_side_by_side_image(img_left, img_right, gap_width=5):
    """Create side-by-side image with a gap in between"""
    # Resize images to match
    img_left, img_right = resize_image_to_match(img_left, img_right)
    
    height = img_left.shape[0]
    width_left = img_left.shape[1]
    width_right = img_right.shape[1]
    
    # Create the combined image with gap
    combined_width = width_left + width_right + gap_width
    combined_img = np.zeros((height, combined_width, 3), dtype=np.uint8)
    
    # Add white gap
    combined_img[:, width_left:width_left + gap_width] = 255
    
    # Place images
    combined_img[:, :width_left] = img_left
    combined_img[:, width_left + gap_width:] = img_right
    
    return combined_img

def add_labels(img, left_label="GT", right_label="Render"):
    """Add labels to the top of the image"""
    height, width = img.shape[:2]
    
    # Calculate label positions
    left_pos = (20, 30)
    right_pos = (width // 2 + 20, 30)
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (0, 0, 255)  # Red color
    thickness = 2
    
    cv2.putText(img, left_label, left_pos, font, font_scale, color, thickness)
    cv2.putText(img, right_label, right_pos, font, font_scale, color, thickness)
    
    return img

def main():
    parser = argparse.ArgumentParser(description='Create comparison video from GT and rendered images')
    parser.add_argument('--gt_dir', default='gt', help='Directory containing GT images')
    parser.add_argument('--renders_dir', default='renders', help='Directory containing rendered images')
    parser.add_argument('--output', default='comparison_video.mp4', help='Output video filename')
    parser.add_argument('--fps', type=int, default=2, help='Video frame rate')
    parser.add_argument('--gap_width', type=int, default=5, help='Gap width between images in pixels')
    
    args = parser.parse_args()
    
    # Get current script directory
    script_dir = Path(__file__).parent
    gt_dir = script_dir / args.gt_dir
    renders_dir = script_dir / args.renders_dir
    output_path = script_dir / args.output
    
    print(f"GT directory: {gt_dir}")
    print(f"Renders directory: {renders_dir}")
    print(f"Output video: {output_path}")
    
    # Check if directories exist
    if not gt_dir.exists():
        print(f"Error: GT directory {gt_dir} does not exist!")
        return
    
    if not renders_dir.exists():
        print(f"Error: Renders directory {renders_dir} does not exist!")
        return
    
    # Get sorted image files
    gt_files = get_sorted_image_files(gt_dir)
    renders_files = get_sorted_image_files(renders_dir)
    
    print(f"Found {len(gt_files)} GT images and {len(renders_files)} rendered images")
    
    # Find common files by matching exact base names (without extension)
    def get_base_name(filename):
        """Extract base name from filename (filename without extension)"""
        return os.path.splitext(filename)[0]
    
    # Create mappings from base name to full filename
    gt_map = {get_base_name(f): f for f in gt_files}
    renders_map = {get_base_name(f): f for f in renders_files}
    
    # Find common base names (exact matches)
    common_base_names = set(gt_map.keys()) & set(renders_map.keys())
    
    # Create list of (gt_file, render_file) pairs
    file_pairs = [(gt_map[base], renders_map[base]) for base in common_base_names]
    
    # Sort pairs by timestamp for consistent ordering
    def extract_timestamp(filename):
        name_part = os.path.splitext(filename)[0]
        if '+' in name_part:
            return int(name_part.split('+')[0].split('-')[-1])
        elif '-' in name_part:
            parts = name_part.split('-')
            for part in reversed(parts):
                if part.isdigit():
                    return int(part)
        import re
        numbers = re.findall(r'\d+', name_part)
        return int(numbers[0]) if numbers else 0
    
    file_pairs.sort(key=lambda x: extract_timestamp(x[0]))
    
    print(f"Example GT base names: {list(gt_map.keys())[:3]}")
    print(f"Example renders base names: {list(renders_map.keys())[:3]}")
    print(f"Common base names found: {len(common_base_names)}")
    
    # Show first few pairs for verification
    print("\nFirst 3 file pairs:")
    for i, (gt_file, render_file) in enumerate(file_pairs[:3]):
        print(f"  {i+1}. GT: {gt_file} <-> Render: {render_file}")
    
    print(f"\nProcessing {len(file_pairs)} common images")
    
    if len(file_pairs) == 0:
        print("Error: No common images found!")
        print(f"GT files sample: {gt_files[:5]}")
        print(f"Render files sample: {renders_files[:5]}")
        return
    
    # Initialize video writer
    video_writer = None
    
    try:
        for i, (gt_filename, render_filename) in enumerate(tqdm(file_pairs, desc="Processing images")):
            # Load images
            gt_path = gt_dir / gt_filename
            render_path = renders_dir / render_filename
            
            gt_img = cv2.imread(str(gt_path))
            render_img = cv2.imread(str(render_path))
            
            if gt_img is None:
                print(f"Warning: Could not load GT image {gt_path}")
                continue
            
            if render_img is None:
                print(f"Warning: Could not load render image {render_path}")
                continue
            
            # Create side-by-side image
            combined_img = create_side_by_side_image(gt_img, render_img, args.gap_width)
            
            # Add labels
            combined_img = add_labels(combined_img)
            
            # Initialize video writer with first frame
            if video_writer is None:
                height, width = combined_img.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(str(output_path), fourcc, args.fps, (width, height))
                print(f"Video dimensions: {width}x{height}")
            
            # Write frame to video
            video_writer.write(combined_img)
    
    finally:
        if video_writer is not None:
            video_writer.release()
    
    print(f"Video created successfully: {output_path}")
    print(f"Total frames: {len(file_pairs)}")
    print(f"Video duration: {len(file_pairs) / args.fps:.2f} seconds")

if __name__ == "__main__":
    main()
