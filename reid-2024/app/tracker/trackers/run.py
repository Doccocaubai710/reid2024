import os
import cv2
import numpy as np
import argparse
import time
from pathlib import Path

# Import TrackTrack (assuming it's in the path or you've installed it)
# These are the expected imports based on typical tracker organization
from app.tracker.trackers.tracker import Tracker  # Main tracker class from TrackTrack repo

def process_video(video_path, output_dir, detector_type="yolov8"):
    """
    Process a video file using TrackTrack
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save results
        detector_type: Type of detector to use
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize detector (placeholder - use your actual detector)
    if detector_type == "yolov8":
        # Initialize YOLOv8 detector
        try:
            from ultralytics import YOLO
            detector = YOLO("yolov8n.pt")
        except ImportError:
            print("YOLOv8 not found. Please install with: pip install ultralytics")
            return
    else:
        print(f"Detector type {detector_type} not supported")
        return
    
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Initialize output video
    output_path = os.path.join(output_dir, Path(video_path).stem + '_tracked.mp4')
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    
    # Initialize tracker
    tracker = Tracker(
        det_thresh=0.5,
        max_age=30,
        min_hits=3,
        iou_threshold=0.3
    )
    
    # Initialize results file
    results_path = os.path.join(output_dir, Path(video_path).stem + '_results.txt')
    results_file = open(results_path, 'w')
    
    frame_id = 0
    processing_times = []
    
    print(f"Processing video: {video_path}")
    print(f"Output will be saved to: {output_path}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        start_time = time.time()
        
        # 1. Detect objects
        detections = detector(frame)[0]  # Use your detector
        
        # 2. Format detections for TrackTrack
        # Expected format: List of [x, y, w, h, score]
        formatted_detections = []
        high_conf_detections = []  # Detections with high confidence
        
        if len(detections.boxes) > 0:
            for det in detections.boxes:
                if det.cls == 0:  # Only track persons (class 0 in COCO)
                    # Get detection info
                    x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
                    confidence = float(det.conf[0])
                    
                    # Convert to [x, y, w, h]
                    detection = [x1, y1, x2-x1, y2-y1, confidence]
                    formatted_detections.append(detection)
                    
                    # Separate high confidence detections (for track initialization)
                    if confidence > 0.6:
                        high_conf_detections.append(detection)
        
        formatted_detections = np.array(formatted_detections)
        high_conf_detections = np.array(high_conf_detections)
        
        # 3. Perform tracking
        if len(formatted_detections) > 0:
            # Call tracker with all detections and high-confidence detections
            tracking_results = tracker.update(formatted_detections, high_conf_detections)
        else:
            tracking_results = []
        
        # 4. Process tracking results
        end_time = time.time()
        processing_times.append(end_time - start_time)
        
        # 5. Visualize results
        for track in tracking_results:
            # Get track information
            track_id = track.track_id
            bbox = track.tlwh  # [x, y, w, h]
            
            # Draw bounding box
            x, y, w, h = map(int, bbox)
            color = (0, 255, 0)  # Green
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"ID: {track_id}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, color, 2)
            
            # Write to results file
            # Format: [frame_id, track_id, x, y, w, h, score]
            results_file.write(f"{frame_id},{track_id},{x},{y},{w},{h},{track.score:.6f}\n")
        
        # Add frame info
        cv2.putText(frame, f"Frame: {frame_id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0, 0, 255), 2)
        
        # Write frame to output video
        out.write(frame)
        
        # Display frame
        cv2.imshow('Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        frame_id += 1
        
        # Print progress every 100 frames
        if frame_id % 100 == 0:
            print(f"Processed {frame_id} frames, FPS: {1.0/np.mean(processing_times[-100:]):.2f}")
    
    # Clean up
    cap.release()
    out.release()
    results_file.close()
    cv2.destroyAllWindows()
    
    # Print summary
    avg_fps = 1.0 / np.mean(processing_times)
    print(f"Processing complete!")
    print(f"Processed {frame_id} frames at {avg_fps:.2f} FPS")
    print(f"Results saved to {results_path}")
    print(f"Video saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video with TrackTrack")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    parser.add_argument("--detector", type=str, default="yolov8", help="Detector type (yolov8)")
    
    args = parser.parse_args()
    process_video(args.video, args.output, args.detector)