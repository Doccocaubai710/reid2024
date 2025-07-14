import os
import logging
import json
from collections import defaultdict

def load_id_switch_detections_from_log(log_file):
    """
    Parse log file to extract ID switch detections
    
    Args:
        log_file: Path to log file containing BYTETracker output
        
    Returns:
        Dictionary mapping track IDs to lists of switch frames
    """
    switch_candidates = defaultdict(list)
    
    if not os.path.exists(log_file):
        logging.warning(f"Log file not found: {log_file}")
        return {}
    
    # Search for ID switch detection logs
    with open(log_file, 'r') as f:
        for line in f:
            if "ID switch detected: Track" in line:
                # Parse the line to extract track ID, frame, and edge
                try:
                    # Format: "ID switch detected: Track X at frame Y near Z edge"
                    parts = line.strip().split()
                    track_id = int(parts[4])
                    frame = int(parts[7])
                    edge = parts[9]
                    
                    switch_candidates[track_id].append({
                        'frame': frame,
                        'edge': edge
                    })
                    
                    logging.info(f"Loaded switch detection for track {track_id} at frame {frame}")
                except (IndexError, ValueError) as e:
                    logging.warning(f"Failed to parse switch detection line: {line.strip()}, error: {e}")
    
    logging.info(f"Loaded {len(switch_candidates)} tracks with ID switch detections")
    return dict(switch_candidates)

def split_tracklet_at_switches(tracklet, switch_frames, min_segment_length=5):
    """
    Split a tracklet at specified switch points
    
    Args:
        tracklet: Tracklet object to split
        switch_frames: List of frame numbers where switches occur
        min_segment_length: Minimum required length for resulting segments
        
    Returns:
        List of split tracklets
    """
    if not switch_frames or len(tracklet.frames) < 2 * min_segment_length:
        return [tracklet]
    
    # Find indices in the tracklet corresponding to switch frames
    switch_indices = []
    for frame in switch_frames:
        try:
            idx = tracklet.frames.index(frame)
            # Ensure sufficient frames in each segment
            if idx >= min_segment_length and len(tracklet.frames) - idx >= min_segment_length:
                switch_indices.append(idx)
        except ValueError:
            # Frame not in this tracklet
            continue
    
    # If no valid switch points found, return original
    if not switch_indices:
        return [tracklet]
    
    # Sort switch indices
    switch_indices.sort()
    
    # Split the tracklet
    result_tracklets = []
    start_idx = 0
    
    for i, split_idx in enumerate(switch_indices):
        # Create new tracklet for this segment
        new_tracklet = type(tracklet)(f"{tracklet.track_id}_part{i}", 
                                    tracklet.camera_id)
        
        # Add frames from start_idx to split_idx
        for j in range(start_idx, split_idx):
            new_tracklet.add_detection(
                tracklet.frames[j],
                tracklet.bboxes[j],
                tracklet.features[j],
                tracklet.scores[j]
            )
        
        # Add if sufficient length
        if len(new_tracklet.frames) >= min_segment_length:
            result_tracklets.append(new_tracklet)
        
        # Move to next segment
        start_idx = split_idx
    
    # Add final segment
    final_tracklet = type(tracklet)(f"{tracklet.track_id}_part{len(result_tracklets)}", 
                                  tracklet.camera_id)
    
    for j in range(start_idx, len(tracklet.frames)):
        final_tracklet.add_detection(
            tracklet.frames[j],
            tracklet.bboxes[j],
            tracklet.features[j],
            tracklet.scores[j]
        )
    
    # Add if sufficient length
    if len(final_tracklet.frames) >= min_segment_length:
        result_tracklets.append(final_tracklet)
    
    return result_tracklets

def apply_id_switch_corrections(tracklets, switch_detections, min_segment_length=5):
    """
    Apply ID switch corrections to a set of tracklets
    
    Args:
        tracklets: Dictionary or list of tracklets
        switch_detections: Dictionary mapping track IDs to switch frames
        min_segment_length: Minimum required length for resulting segments
        
    Returns:
        List of corrected tracklets
    """
    # Convert to list if needed
    tracklets_list = list(tracklets.values()) if isinstance(tracklets, dict) else tracklets
    
    corrected_tracklets = []
    split_count = 0
    
    for tracklet in tracklets_list:
        track_id = tracklet.track_id
        
        # Check if this tracklet has switch detections
        if track_id in switch_detections:
            switch_frames = [info['frame'] for info in switch_detections[track_id]]
            split_results = split_tracklet_at_switches(
                tracklet, 
                switch_frames, 
                min_segment_length
            )
            
            if len(split_results) > 1:
                logging.info(f"Split track {track_id} into {len(split_results)} parts")
                split_count += 1
                corrected_tracklets.extend(split_results)
            else:
                # No valid splits, keep original
                corrected_tracklets.append(tracklet)
        else:
            # No switch detections, keep original
            corrected_tracklets.append(tracklet)
    
    logging.info(f"Applied ID switch corrections to {split_count} tracklets")
    return corrected_tracklets

def process_tracklets_with_switch_detection(tracklets, log_file=None, switch_file=None, 
                                           min_segment_length=5, save_results=True):
    """
    Process tracklets with ID switch detection from logs or saved switch file
    
    Args:
        tracklets: Dictionary or list of tracklets to process
        log_file: Path to log file with switch detections
        switch_file: Path to JSON file with saved switch detections
        min_segment_length: Minimum required length for resulting segments
        save_results: Whether to save the switch detections to a file
        
    Returns:
        List of corrected tracklets
    """
    switch_detections = {}
    
    # Try to load switch detections from switch file
    if switch_file and os.path.exists(switch_file):
        try:
            with open(switch_file, 'r') as f:
                switch_detections = json.load(f)
                # Convert string keys back to integers
                switch_detections = {int(k): v for k, v in switch_detections.items()}
            logging.info(f"Loaded switch detections from {switch_file}")
        except Exception as e:
            logging.warning(f"Failed to load switch file {switch_file}: {e}")
    
    # If no switch file or loading failed, try log file
    if not switch_detections and log_file:
        switch_detections = load_id_switch_detections_from_log(log_file)
        
        # Save to file if requested
        if save_results and switch_detections:
            try:
                switch_save_path = switch_file or "id_switch_detections.json"
                with open(switch_save_path, 'w') as f:
                    # Convert int keys to strings for JSON serialization
                    json_data = {str(k): v for k, v in switch_detections.items()}
                    json.dump(json_data, f, indent=2)
                logging.info(f"Saved switch detections to {switch_save_path}")
            except Exception as e:
                logging.warning(f"Failed to save switch detections: {e}")
    
    # Apply corrections
    if switch_detections:
        return apply_id_switch_corrections(tracklets, switch_detections, min_segment_length)
    else:
        logging.warning("No ID switch detections found, returning original tracklets")
        return tracklets if isinstance(tracklets, list) else list(tracklets.values())

# Example usage in an offline processing pipeline
def detect_id_switches_from_log(tracklets, log_file="example.log", switch_file=None):
    """
    Wrapper function to detect ID switches from log files and apply to tracklets
    
    Args:
        tracklets: Dictionary or list of tracklets
        log_file: Path to log file with switch detections
        switch_file: Optional path to save/load switch detections
        
    Returns:
        List of corrected tracklets
    """
    return process_tracklets_with_switch_detection(
        tracklets, 
        log_file=log_file, 
        switch_file=switch_file
    )