import os
import numpy as np
import cv2
import copy
import logging
import torch
import torch.nn as nn
from scipy.spatial.distance import cdist, cosine
from sklearn.cluster import AgglomerativeClustering
from define_zone import define_reliable_uncertain_zones, filter_tracklets_by_zone
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("r_matching.log"),
        logging.StreamHandler()
    ]
)

def calculate_err(first_dist, second_dist):
    """
    Calculate error/difference between first and second best matches.
    This helps determine if the match is confident or uncertain.
    
    Args:
        first_dist: Distance to best match
        second_dist: Distance to second best match
        
    Returns:
        Error measure indicating confidence in the match
    """
    # Implementation based on the original paper's approach
    # Higher value means more confident match
    if second_dist == 0:
        return 100  # Very confident if no second candidate
    
    # Calculate ratio between first and second distances
    # (similar to Lowe's ratio test in SIFT)
    return 100 * (1 - (first_dist / second_dist))

def check_time_overlap(tracklet1, tracklet2):
    """
    Check if two tracklets overlap in time (share any frames).
    
    Args:
        tracklet1, tracklet2: Tracklet objects with frames property
        
    Returns:
        True if tracklets overlap in time, False otherwise
    """
    frames1 = set(tracklet1.frames)
    frames2 = set(tracklet2.frames)
    return len(frames1.intersection(frames2)) > 0

def find_anchor_frame(tracklets):
    """
    Find the frame with the most people (highest density).
    
    Args:
        tracklets: List of tracklet objects
        
    Returns:
        Frame number with highest person count
    """
    # Count people per frame
    frame_counts = {}
    for tracklet in tracklets:
        for frame in tracklet.frames:
            if frame not in frame_counts:
                frame_counts[frame] = 0
            frame_counts[frame] += 1
    
    # Find frame with most people
    if not frame_counts:
        return 0  # Default to frame 0 if no data
        
    anchor_frame = max(frame_counts.items(), key=lambda x: x[1])[0]
    logging.info(f"Selected anchor frame {anchor_frame} with {frame_counts[anchor_frame]} people")
    return anchor_frame

def get_initial_clusters(tracklets, anchor_frame):
    """
    Create initial clusters from tracklets visible in the anchor frame.
    
    Args:
        tracklets: List of tracklet objects
        anchor_frame: Frame number for initialization
        
    Returns:
        Dictionary of cluster_id -> list of tracklets
    """
    clusters = {}
    cluster_idx = 0
    
    for tracklet in tracklets:
        if anchor_frame in tracklet.frames:
            clusters[cluster_idx] = [tracklet]
            cluster_idx += 1
    
    logging.info(f"Created {len(clusters)} initial clusters from anchor frame {anchor_frame}")
    return clusters

def update_mean_features(tracklets_in_cluster):
    """
    Calculate mean feature vector for a cluster of tracklets.
    
    Args:
        tracklets_in_cluster: List of tracklet objects in a cluster
        
    Returns:
        Mean feature vector for the cluster
    """
    if not tracklets_in_cluster:
        return None
        
    # Extract features from each tracklet
    features = []
    for tracklet in tracklets_in_cluster:
        if hasattr(tracklet, 'mean_features') and callable(getattr(tracklet, 'mean_features')):
            # If tracklet has mean_features method, use it
            feat = tracklet.mean_features()
            if feat is not None:
                features.append(feat)
        elif hasattr(tracklet, 'features') and tracklet.features:
            # Otherwise calculate mean from features array
            feat = np.mean(tracklet.features, axis=0)
            features.append(feat)
    
    if not features:
        return None
        
    # Calculate mean across all tracklets
    mean_feature = np.mean(features, axis=0)
    return mean_feature

def r_matching(tracklets, frame_shape, similarity_threshold=0.2, err_threshold=25):
    """
    Implement R-matching algorithm for single-camera tracking.
    
    Args:
        tracklets: List of tracklet objects
        frame_shape: Tuple of (height, width) for frame dimensions
        similarity_threshold: Maximum cosine distance for valid matches
        err_threshold: Minimum error value for confident matches
        
    Returns:
        matched_clusters: Dictionary mapping cluster IDs to lists of tracklets
        uncertain_tracklets: List of tracklets that couldn't be confidently matched
    """
    # 1. Define reliable and uncertain zones
    
    reliable_zone, _ = define_reliable_uncertain_zones(frame_shape, visualize=False)
    
    # 2. Split tracklets into reliable and uncertain
    reliable_tracklets, uncertain_tracklets = filter_tracklets_by_zone(tracklets, reliable_zone)
    logging.info(f"Initial split: {len(reliable_tracklets)} reliable, {len(uncertain_tracklets)} uncertain tracklets")
    
    # 3. Find anchor frame with most people
    anchor_frame = find_anchor_frame(reliable_tracklets)
    
    # 4. Create initial clusters from tracklets in anchor frame
    clusters = get_initial_clusters(reliable_tracklets, anchor_frame)
    
    # If no clusters were created, try using all tracklets
    if not clusters:
        logging.warning("No tracklets found in anchor frame. Using all tracklets.")
        anchor_frame = find_anchor_frame(tracklets)
        clusters = get_initial_clusters(tracklets, anchor_frame)
    
    # 5. Split remaining tracklets into prefix and postfix
    prefix_tracklets = []
    postfix_tracklets = []
    
    # Exclude tracklets already in clusters
    clustered_ids = [t.track_id for cluster in clusters.values() for t in cluster]
    
    for tracklet in reliable_tracklets:
        if tracklet.track_id in clustered_ids:
            continue
            
        if max(tracklet.frames) < anchor_frame:
            prefix_tracklets.append(tracklet)
        else:
            postfix_tracklets.append(tracklet)
    
    logging.info(f"Split remaining tracklets: {len(prefix_tracklets)} prefix, {len(postfix_tracklets)} postfix")
    
    # 6. Sort tracklets chronologically
    prefix_tracklets.sort(key=lambda x: max(x.frames), reverse=True)  # Sort by last frame, descending
    postfix_tracklets.sort(key=lambda x: min(x.frames))  # Sort by first frame, ascending
    
    # 7. Process prefix tracklets
    prefix_uncertain = []
    matched_clusters = clusters
    
    for tracklet in prefix_tracklets:
        # Find viable clusters (no time overlap)
        viable_clusters = []
        
        for cluster_id, cluster_tracklets in matched_clusters.items():
            # Check if tracklet overlaps with any in the cluster
            overlap = False
            for cluster_tracklet in cluster_tracklets:
                if check_time_overlap(tracklet, cluster_tracklet):
                    overlap = True
                    break
            
            if not overlap:
                # Calculate mean feature for cluster
                cluster_feature = update_mean_features(cluster_tracklets)
                
                # Skip if no valid feature
                if cluster_feature is None:
                    continue
                
                # Calculate cosine distance
                tracklet_feature = tracklet.mean_features()
                distance = cosine(tracklet_feature, cluster_feature)
                
                viable_clusters.append((distance, cluster_id))
        
        # If no viable clusters, mark as uncertain
        if not viable_clusters:
            prefix_uncertain.append(tracklet)
            continue
        
        # Sort by distance (ascending)
        viable_clusters.sort(key=lambda x: x[0])
        
        # Check if best match is good enough
        if viable_clusters[0][0] > similarity_threshold:
            prefix_uncertain.append(tracklet)
            continue
        
        # If multiple candidates, check confidence
        if len(viable_clusters) > 1:
            first_dist = viable_clusters[0][0]
            second_dist = viable_clusters[1][0]
            
            error = calculate_err(first_dist, second_dist)
            
            # If not confident enough, mark as uncertain
            if error < err_threshold:
                prefix_uncertain.append(tracklet)
                continue
        
        # Add to best matching cluster
        best_cluster_id = viable_clusters[0][1]
        matched_clusters[best_cluster_id].append(tracklet)
        
        # Update cluster feature
        # (Not strictly necessary as we recalculate for each tracklet,
        # but can improve results for subsequent matches)
        
    logging.info(f"After prefix matching: {len(prefix_uncertain)} uncertain tracklets")
    
    # 8. Process postfix tracklets (similar to prefix)
    postfix_uncertain = []
    
    for tracklet in postfix_tracklets:
        # Find viable clusters (no time overlap)
        viable_clusters = []
        
        for cluster_id, cluster_tracklets in matched_clusters.items():
            # Check if tracklet overlaps with any in the cluster
            overlap = False
            for cluster_tracklet in cluster_tracklets:
                if check_time_overlap(tracklet, cluster_tracklet):
                    overlap = True
                    break
            
            if not overlap:
                # Calculate mean feature for cluster
                cluster_feature = update_mean_features(cluster_tracklets)
                
                # Skip if no valid feature
                if cluster_feature is None:
                    continue
                
                # Calculate cosine distance
                tracklet_feature = tracklet.mean_features()
                distance = cosine(tracklet_feature, cluster_feature)
                
                viable_clusters.append((distance, cluster_id))
        
        # If no viable clusters, mark as uncertain
        if not viable_clusters:
            postfix_uncertain.append(tracklet)
            continue
        
        # Sort by distance (ascending)
        viable_clusters.sort(key=lambda x: x[0])
        
        # Check if best match is good enough
        if viable_clusters[0][0] > similarity_threshold:
            postfix_uncertain.append(tracklet)
            continue
        
        # If multiple candidates, check confidence
        if len(viable_clusters) > 1:
            first_dist = viable_clusters[0][0]
            second_dist = viable_clusters[1][0]
            
            error = calculate_err(first_dist, second_dist)
            
            # If not confident enough, mark as uncertain
            if error < err_threshold:
                postfix_uncertain.append(tracklet)
                continue
        
        # Add to best matching cluster
        best_cluster_id = viable_clusters[0][1]
        matched_clusters[best_cluster_id].append(tracklet)
    
    logging.info(f"After postfix matching: {len(postfix_uncertain)} uncertain tracklets")
    
    # 9. Combine all uncertain tracklets
    all_uncertain = prefix_uncertain + postfix_uncertain + uncertain_tracklets
    
    # 10. Match remaining uncertain tracklets to clusters as a separate step
    matched_uncertain = []
    final_uncertain = []
    
    for tracklet in all_uncertain:
        # Similar to above, but with relaxed constraints
        viable_clusters = []
        
        for cluster_id, cluster_tracklets in matched_clusters.items():
            # Skip time overlap check for uncertain tracklets
            # to prioritize appearance similarity
            
            # Calculate mean feature for cluster
            cluster_feature = update_mean_features(cluster_tracklets)
            
            # Skip if no valid feature
            if cluster_feature is None:
                continue
            
            # Calculate cosine distance
            tracklet_feature = tracklet.mean_features()
            distance = cosine(tracklet_feature, cluster_feature)
            
            viable_clusters.append((distance, cluster_id))
        
        # If no viable clusters, keep as uncertain
        if not viable_clusters:
            final_uncertain.append(tracklet)
            continue
        
        # Sort by distance (ascending)
        viable_clusters.sort(key=lambda x: x[0])
        
        # Use a more relaxed threshold for uncertain tracklets
        relaxed_threshold = similarity_threshold * 1.5
        
        # Check if best match is good enough
        if viable_clusters[0][0] > relaxed_threshold:
            final_uncertain.append(tracklet)
            continue
        
        # Add to best matching cluster
        best_cluster_id = viable_clusters[0][1]
        matched_clusters[best_cluster_id].append(tracklet)
        matched_uncertain.append(tracklet)
    
    logging.info(f"Final matching: {len(matched_uncertain)} uncertain tracklets matched, {len(final_uncertain)} remain uncertain")
    
    # Return matched clusters and remaining uncertain tracklets
    return matched_clusters, final_uncertain

def integrate_r_matching(tracklets, frame_shape, err_threshold=23):
    """
    Integration function for using R-matching in your tracking system.
    
    Args:
        tracklets: Dictionary of track_id -> tracklet
        frame_shape: Tuple of (height, width)
        err_threshold: Threshold for confident matches
        
    Returns:
        track_to_cluster: Dictionary mapping track_id to cluster_id
    """
    # Convert dictionary to list for R-matching
    tracklet_list = list(tracklets.values())
    
    # Adjust threshold based on camera if needed
    camera_id = tracklet_list[0].camera_id if tracklet_list else "unknown"
    if "c005" in str(camera_id):
        err_threshold = 27
    
    # Run R-matching
    matched_clusters, uncertain_tracklets = r_matching(
        tracklet_list, 
        frame_shape,
        similarity_threshold=0.2,  # Adjust based on testing
        err_threshold=err_threshold
    )
    
    # Create mapping from track_id to cluster_id
    track_to_cluster = {}
    
    # Process matched clusters
    for cluster_id, cluster_tracklets in matched_clusters.items():
        for tracklet in cluster_tracklets:
            track_to_cluster[tracklet.track_id] = cluster_id
    
    # Assign new cluster IDs to uncertain tracklets
    next_cluster_id = max(matched_clusters.keys()) + 1 if matched_clusters else 0
    
    for tracklet in uncertain_tracklets:
        track_to_cluster[tracklet.track_id] = next_cluster_id
        next_cluster_id += 1
    
    return track_to_cluster

# Example usage in offline_processing.py
def improved_process_session(session_dir):
    """
    Process a session with improved R-matching.
    
    Args:
        session_dir: Path to session directory
    """
    # Load tracklets
    tracklets = load_data(...)
    
    # Get frame dimensions
    # (you'll need to extract this from your video or tracking data)
    frame_shape = (1080, 1920)  # Example values - update with actual dimensions
    
    # Detect and correct ID switches
    corrected_tracklets = detect_id_switches(tracklets)
    
    # Use R-matching for single-camera matching
    track_to_cluster = integrate_r_matching(corrected_tracklets, frame_shape)
    
    # Save results
    save_results(corrected_tracklets, track_to_cluster, output_file)