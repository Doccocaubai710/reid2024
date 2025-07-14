import os
import argparse
import logging
import numpy as np
import cv2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("visualization.log"),
        logging.StreamHandler()
    ]
)

def draw_detection(img, bboxes, scores, ids, mask_alpha=0.3):
    """Draws detection boxes on image with scores and IDs"""
    height, width = img.shape[:2]
    np.random.seed(0)
    rng = np.random.default_rng(3)
    colors = rng.uniform(0, 255, size=(1, 3))
    mask_img = img.copy()
    det_img = img.copy()
    size = min([height, width]) * 0.0006
    text_thickness = int(min([height, width]) * 0.001)
    
    for bbox, score, id_ in zip(bboxes, scores, ids):
        color = colors[0]
        bbox = np.array(bbox)
        x1, y1, x2, y2 = bbox.astype(np.int64)
        # Draw rectangle
        cv2.rectangle(det_img, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)
        caption = f"body {int(score*100)}% ID: {id_}"
        (tw, th), _ = cv2.getTextSize(
            text=caption,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=size,
            thickness=text_thickness,
        )

        th = int(th * 1.2)
        cv2.rectangle(det_img, (x1, y1), (x1 + tw, y1 - th), color, -1)
        cv2.rectangle(mask_img, (x1, y1), (x1 + tw, y1 - th), color, -1)
        cv2.putText(
            det_img,
            caption,
            (x1, y1),
            cv2.FONT_HERSHEY_SIMPLEX,
            size,
            (255, 255, 255),
            text_thickness,
            cv2.LINE_AA,
        )
        cv2.putText(
            mask_img,
            caption,
            (x1, y1),
            cv2.FONT_HERSHEY_SIMPLEX,
            size,
            (255, 255, 255),
            text_thickness,
            cv2.LINE_AA,
        )
    return cv2.addWeighted(mask_img, mask_alpha, det_img, 1 - mask_alpha, 0)

def visualize_tracking(input_video, tracking_file, output_video):
    """
    Visualize tracking results on the input video
    
    Args:
        input_video: Path to input video file
        tracking_file: Path to tracking results file
        output_video: Path to output video
    """
    # Read tracking results
    tracking_data = {}
    with open(tracking_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 8:  # Ensure there are enough parts
                logging.warning(f"Skipping invalid line: {line}")
                continue
                
            # Parse tracking data format:
            # 2025-05-11 18-58-12-f186221692dc492f9a2601485b29f949,2,83,1038.7554178843957,198.0,1164.8445821156042,536.6,-1,-1
            # The format is: timestamp,id,frame,x1,y1,x2,y2,-1,-1
            
            try:
                # Skip the timestamp (parts[0])
                person_id = int(parts[1])
                frame_num = int(parts[2])
                x1 = float(parts[3])
                y1 = float(parts[4])
                x2 = float(parts[5])
                y2 = float(parts[6])
                
                # Using confidence score of 1.0 since it's not in the data
                score = 1.0
                
                # Store bounding box in format [x1, y1, x2, y2]
                bbox = [x1, y1, x2, y2]
                
                if frame_num not in tracking_data:
                    tracking_data[frame_num] = []
                    
                tracking_data[frame_num].append((person_id, bbox, score))
            except ValueError as e:
                logging.warning(f"Error parsing line: {line}, Error: {e}")
                continue
    
    if not tracking_data:
        logging.error(f"No valid tracking data found in {tracking_file}")
        return
    
    # Open input video
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        logging.error(f"Could not open video: {input_video}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logging.info(f"Video properties: {width}x{height} @ {fps}fps, {frame_count} frames")
    logging.info(f"Tracking data covers {len(tracking_data)} frames")
    
    # Create output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v codec
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    # Process video frames
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Draw tracking results if available for this frame
        if frame_idx in tracking_data:
            bboxes = []
            scores = []
            ids = []
            
            for person_id, bbox, score in tracking_data[frame_idx]:
                bboxes.append(bbox)
                scores.append(score)
                ids.append(person_id)
            
            # Draw bounding boxes
            if bboxes:
                frame = draw_detection(frame, bboxes, scores, ids)
        
        # Add frame number to each frame
        cv2.putText(
            frame, 
            f"Frame: {frame_idx}", 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 255, 0), 
            2, 
            cv2.LINE_AA
        )
        
        # Write frame to output video
        out.write(frame)
        
        # Show progress
        if frame_idx % 100 == 0:
            logging.info(f"Processed {frame_idx}/{frame_count} frames")
        
        frame_idx += 1
    
    # Release resources
    cap.release()
    out.release()
    
    logging.info(f"Visualization completed. Output saved to {output_video}")

def main():
    parser = argparse.ArgumentParser(description="Visualize tracking results on a video")
    parser.add_argument("--input_video", required=True, help="Path to input video file")
    parser.add_argument("--tracking_file", required=True, help="Path to tracking results file")
    parser.add_argument("--output_video", required=True, help="Path to output video")
    args = parser.parse_args()
    
    if not os.path.exists(args.input_video):
        logging.error(f"Input video not found: {args.input_video}")
        return
        
    if not os.path.exists(args.tracking_file):
        logging.error(f"Tracking file not found: {args.tracking_file}")
        return
    
    visualize_tracking(args.input_video, args.tracking_file, args.output_video)

if __name__ == "__main__":
    main()