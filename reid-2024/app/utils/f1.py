def extract_all_selected_bboxes(
    tracking_file,
    video_path,
    output_dir,
    selection_text,
    track_id_mapping=None,  # Add mapping parameter
    margin=10
):
    """
    Extract bounding boxes for selected track IDs and frames.
    
    Args:
        tracking_file: Path to tracking results file
        video_path: Path to the source video
        output_dir: Directory to save the output images
        selection_text: Text with track IDs and selected frames
        track_id_mapping: Dictionary mapping requested IDs to tracking file IDs, or None
        margin: Margin to add around the bounding box
    """
    import cv2
    import os
    import re
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Parse the selection text to extract track IDs and frames
    requested_track_frame_dict = {}
    pattern = r'(\d+) has selected frames (\d+)'
    matches = re.findall(pattern, selection_text)
    
    for track_id_str, frame_id_str in matches:
        track_id = int(track_id_str)
        frame_id = int(frame_id_str)
        requested_track_frame_dict[track_id] = frame_id
    
    print(f"Parsed {len(requested_track_frame_dict)} track ID and frame pairs")
    
    # Handle the ID mapping
    if track_id_mapping is None:
        # Create default mapping (1:1)
        track_id_mapping = {
            # These two specific mappings from the example
            28: 3,   # Track 28 in list maps to track 3 in tracking file
            29: 12,  # Track 29 in list maps to track 12 in tracking file
        }
        
        # For all other IDs, map them to themselves by default
        for track_id in requested_track_frame_dict.keys():
            if track_id not in track_id_mapping:
                track_id_mapping[track_id] = track_id
    
    # Create reverse mapping for lookup
    track_frame_to_requested_id = {}
    for requested_id, frame_id in requested_track_frame_dict.items():
        tracking_id = track_id_mapping.get(requested_id, requested_id)
        track_frame_to_requested_id[(tracking_id, frame_id)] = requested_id
    
    print("Using track ID mapping:")
    for requested_id, tracking_id in sorted(track_id_mapping.items()):
        if requested_id != tracking_id:
            print(f"  Requested ID {requested_id} -> Tracking file ID {tracking_id}")
    
    # Read tracking results to find all detections
    detections_by_track_frame = {}
    
    print(f"Reading tracking data from {tracking_file}")
    with open(tracking_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 6:  # Need at least frame_id,track_id,x,y,w,h
                continue
            
            try:
                # Format: frame_id,track_id,x,y,x2,y2,score
                frame_id = int(float(parts[0]))
                track_id = int(float(parts[1]))
                
                # Check if this is one of our target track-frame pairs
                track_frame_pair = (track_id, frame_id)
                if track_frame_pair in track_frame_to_requested_id:
                    requested_id = track_frame_to_requested_id[track_frame_pair]
                    
                    x = float(parts[2])
                    y = float(parts[3])
                    x2 = float(parts[4])
                    y2 = float(parts[5])
                    
                    w = x2 - x
                    h = y2 - y
                    
                    confidence = float(parts[6]) if len(parts) > 6 else 1.0
                    
                    detections_by_track_frame[(requested_id, frame_id)] = {
                        'tracking_id': track_id,
                        'x1': x, 'y1': y,
                        'x2': x2, 'y2': y2,
                        'width': w, 'height': h,
                        'confidence': confidence
                    }
            except (ValueError, IndexError) as e:
                print(f"Warning: Could not parse line: {line.strip()} - {e}")
                continue
    
    # Check which track-frame pairs were found
    not_found = []
    for requested_id, frame_id in requested_track_frame_dict.items():
        if (requested_id, frame_id) not in detections_by_track_frame:
            not_found.append((requested_id, frame_id))
    
    if not_found:
        print(f"Warning: Could not find {len(not_found)} track-frame pairs in the tracking file:")
        for requested_id, frame_id in not_found:
            tracking_id = track_id_mapping.get(requested_id, requested_id)
            print(f"  Requested track {requested_id} (tracking ID {tracking_id}), Frame {frame_id}")
    
    print(f"Found {len(detections_by_track_frame)} valid detections out of {len(requested_track_frame_dict)} requested")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video info
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Process each detection
    saved_images = []
    
    for (requested_id, frame_id), detection in detections_by_track_frame.items():
        # Seek to the specific frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        
        if not ret:
            print(f"Warning: Could not read frame {frame_id}")
            continue
        
        # Get bounding box coordinates with margin
        x1 = max(0, int(detection['x1'] - margin))
        y1 = max(0, int(detection['y1'] - margin))
        x2 = min(frame_width, int(detection['x2'] + margin))
        y2 = min(frame_height, int(detection['y2'] + margin))
        
        # Crop the image
        cropped = frame[y1:y2, x1:x2]
        
        if cropped.size == 0:
            print(f"Warning: Empty crop for track {requested_id} at frame {frame_id}")
            continue
        
        # Create output path
        file_path = os.path.join(output_dir, f"track_{requested_id}_frame_{frame_id}.jpg")
        
        # Save the cropped image
        cv2.imwrite(file_path, cropped)
        saved_images.append(file_path)
        
        tracking_id = detection['tracking_id']
        if requested_id != tracking_id:
            print(f"Saved crop for requested track {requested_id} (tracking ID {tracking_id}), frame {frame_id} to {file_path}")
        else:
            print(f"Saved crop for track {requested_id}, frame {frame_id} to {file_path}")
    
    # Release resources
    cap.release()
    
    print(f"\nSummary:")
    print(f"Total track-frame pairs requested: {len(requested_track_frame_dict)}")
    print(f"Total images saved: {len(saved_images)}")
    print(f"All images saved to: {output_dir}")
    
    return saved_images, requested_track_frame_dict
selection_text = """
1 has selected frames 291
2 has selected frames 96
3 has selected frames 336
28 has selected frames 413
4 has selected frames 294
5 has selected frames 371
6 has selected frames 690
7 has selected frames 699
8 has selected frames 716
9 has selected frames 938
10 has selected frames 1258
11 has selected frames 1249
12 has selected frames 1431
29 has selected frames 1519
13 has selected frames 1833
14 has selected frames 1603
15 has selected frames 1734
16 has selected frames 2162
17 has selected frames 2120
18 has selected frames 2549
19 has selected frames 2296
20 has selected frames 2307
21 has selected frames 2443
22 has selected frames 3331
23 has selected frames 3412
24 has selected frames 2885
25 has selected frames 3488
26 has selected frames 3155
27 has selected frames 3578
"""
track_id_mapping = {
    28: 3,   # Track 28 in list maps to track 3 in tracking file
    29: 12,  # Track 29 in list maps to track 12 in tracking file
    # Add any other mappings needed
}
# Extract all the bounding boxes for the selected track IDs and frames
saved_images, track_frame_dict = extract_all_selected_bboxes(
    tracking_file="/home/aidev/workspace/reid/Thesis/reid-2024/trash/2025-05-22 20-42-37-2de4f2989e1f4e4298b81da615925c83/tracking_results.txt",
    video_path="/home/aidev/workspace/reid/Thesis/reid-2024/app/assets/videos/cam2.MOV",
    output_dir="Screenshots/person_crops",
    selection_text=selection_text,
    track_id_mapping=track_id_mapping,
    margin=10
)

# Print the parsed dictionary for reference
print("\nTrack ID to Frame ID mapping:")
for track_id, frame_id in sorted(track_frame_dict.items()):
    print(f"Track ID {track_id} -> Frame {frame_id}")