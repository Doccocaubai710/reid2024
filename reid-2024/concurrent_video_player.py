import cv2
import numpy as np
import os
import argparse
from datetime import datetime

def process_videos(video1_path, video2_path, output_path=None, layout="horizontal", scale=1.0):
    """
    Process two videos concurrently and save the result as MP4
    
    Args:
        video1_path: Path to first video
        video2_path: Path to second video
        output_path: Path to save the output video (default: timestamp-based filename)
        layout: "horizontal", "vertical", or "pip" (picture-in-picture)
        scale: Scale factor for the output video
    """
    # Open video captures
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)
    
    # Check if videos opened successfully
    if not cap1.isOpened() or not cap2.isOpened():
        print("Error: Could not open one or both videos.")
        return
    
    # Get video properties
    width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps1 = cap1.get(cv2.CAP_PROP_FPS)
    fps2 = cap2.get(cv2.CAP_PROP_FPS)
    
    # Use the minimum fps to ensure sync
    fps = min(fps1, fps2)
    
    # Calculate output dimensions based on layout
    if layout == "horizontal":
        out_width = int((width1 + width2) * scale)
        out_height = int(max(height1, height2) * scale)
    elif layout == "vertical":
        out_width = int(max(width1, width2) * scale)
        out_height = int((height1 + height2) * scale)
    elif layout == "pip":
        # Main video with picture-in-picture
        out_width = int(width1 * scale)
        out_height = int(height1 * scale)
    else:
        print(f"Invalid layout: {layout}. Using horizontal.")
        out_width = int((width1 + width2) * scale)
        out_height = int(max(height1, height2) * scale)
        layout = "horizontal"
    
    # Set output path if not provided
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"concurrent_video_{timestamp}.mp4"
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))
    
    # Process frames
    while True:
        # Read frames
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        # Break if either video ends
        if not ret1 or not ret2:
            break
        
        # Resize frames for the output
        frame1 = cv2.resize(frame1, (int(width1 * scale), int(height1 * scale)))
        frame2 = cv2.resize(frame2, (int(width2 * scale), int(height2 * scale)))
        
        # Create combined frame based on layout
        if layout == "horizontal":
            # Pad the shorter video with black
            if frame1.shape[0] < frame2.shape[0]:
                pad = np.zeros((frame2.shape[0] - frame1.shape[0], frame1.shape[1], 3), dtype=np.uint8)
                frame1 = np.vstack((frame1, pad))
            elif frame2.shape[0] < frame1.shape[0]:
                pad = np.zeros((frame1.shape[0] - frame2.shape[0], frame2.shape[1], 3), dtype=np.uint8)
                frame2 = np.vstack((frame2, pad))
            
            combined_frame = np.hstack((frame1, frame2))
            
        elif layout == "vertical":
            # Pad the narrower video with black
            if frame1.shape[1] < frame2.shape[1]:
                pad = np.zeros((frame1.shape[0], frame2.shape[1] - frame1.shape[1], 3), dtype=np.uint8)
                frame1 = np.hstack((frame1, pad))
            elif frame2.shape[1] < frame1.shape[1]:
                pad = np.zeros((frame2.shape[0], frame1.shape[1] - frame2.shape[1], 3), dtype=np.uint8)
                frame2 = np.hstack((frame2, pad))
            
            combined_frame = np.vstack((frame1, frame2))
            
        elif layout == "pip":
            # Calculate picture-in-picture size (1/4 of the original)
            pip_width = frame2.shape[1] // 4
            pip_height = frame2.shape[0] // 4
            
            # Resize second video for PIP
            pip_frame = cv2.resize(frame2, (pip_width, pip_height))
            
            # Create a copy of the first frame
            combined_frame = frame1.copy()
            
            # Place the PIP in the bottom-right corner with 10px margin
            x_offset = combined_frame.shape[1] - pip_width - 10
            y_offset = combined_frame.shape[0] - pip_height - 10
            
            # Overlay PIP on main video
            combined_frame[y_offset:y_offset+pip_height, x_offset:x_offset+pip_width] = pip_frame
        
        # Write the combined frame
        out.write(combined_frame)
    
    # Release resources
    cap1.release()
    cap2.release()
    out.release()
    
    print(f"Concurrent video saved as: {output_path}")
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process two videos concurrently and save as MP4')
    parser.add_argument('video1', help='Path to first video')
    parser.add_argument('video2', help='Path to second video')
    parser.add_argument('--output', '-o', help='Output file path (default: timestamp-based filename)')
    parser.add_argument('--layout', '-l', choices=['horizontal', 'vertical', 'pip'], 
                        default='horizontal', help='Layout for combined video')
    parser.add_argument('--scale', '-s', type=float, default=1.0, 
                        help='Scale factor for output video')
    
    args = parser.parse_args()
    
    process_videos(args.video1, args.video2, args.output, args.layout, args.scale)