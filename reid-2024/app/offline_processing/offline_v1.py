
import os
import argparse
import logging
import json
import numpy as np
import torch
from tqdm import tqdm
import time
from scipy.spatial import distance
from sklearn.mixture import GaussianMixture
import cv2
import copy
from collections import defaultdict
from shapely.geometry import Polygon
from app.offline_processing.id_switch_v1 import Tracklet, extract_id_switches_from_log, filter_nearby_switches
from app.utils.f1_og import ReIDFeatureExtractor
import re
import pickle
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("offline_processing.log"),
        logging.StreamHandler()
    ]
)

def load_data(tracking_file, feature_file):
    """
    Load tracking results and features from files
    
    Args:
        tracking_file: Path to tracking results file
        feature_file: Path to feature vectors file in JSONL format
        
    Returns:
        Dictionary of tracklets
    """
    # Extract session ID from directory name
    session_dir = os.path.dirname(tracking_file)
    session_id = os.path.basename(session_dir)
    
    # Load feature data from JSONL file
    feature_data = {}
    with open(feature_file, 'r') as f:
        for line in f:
            try:
                frame_data = json.loads(line.strip())
                frame_num = frame_data.get('frame')
                if frame_num is not None:
                    feature_data[frame_num] = frame_data.get('detections', [])
            except json.JSONDecodeError:
                logging.warning(f"Skipping invalid JSON line in {feature_file}")
    
    # Initialize tracklets dictionary
    tracklets = {}
    
    # Read tracking file
    with open(tracking_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 7:
                continue
                
            frame = int(parts[0])
            track_id = int(parts[1])
            x1, y1, x2, y2 = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            bbox = [x1, y1, x2, y2]
            score = float(parts[6]) if len(parts) > 6 else 1.0
            
            # Create tracklet if it doesn't exist
            if track_id not in tracklets:
                tracklets[track_id] = Tracklet(track_id, session_id)
            
            # Get feature for this detection from feature file
            feature = None
            if frame in feature_data:
                for detection in feature_data[frame]:
                    if detection.get('track_id') == track_id:
                        feature = np.array(detection.get('feature'))
                        break
                
            # Add detection to tracklet
            tracklets[track_id].add_detection(frame, bbox, feature, score)
    
    logging.info(f"Loaded {len(tracklets)} tracklets for session {session_id}")
    return tracklets
def create_tracklet_distance_matrix(
    tracklets, 
    output_path="distance_matrix.png",
    distance_metric="cosine",
    show_values=True,
    max_tracklets=50,
    cmap="viridis_r",
    figsize=None
):
    """
    Create and visualize a distance matrix between all tracklet features.
    
    Args:
        tracklets: Dictionary of Tracklet objects (key: track_id, value: Tracklet)
        output_path: Path to save the visualization
        distance_metric: Distance metric to use ('cosine', 'euclidean', etc.)
        show_values: Whether to show distance values in the cells
        max_tracklets: Maximum number of tracklets to visualize (for readability)
        cmap: Colormap for the heatmap (default: viridis_r - reversed so darker=closer)
        figsize: Figure size (width, height) in inches, or None for auto-sizing
        
    Returns:
        distance_matrix: Numpy array containing the distance matrix
        track_ids: List of track IDs corresponding to the matrix rows/columns
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.spatial.distance import cdist
    import numpy as np
    import os
    
    # Get list of track IDs and corresponding tracklets
    track_ids = list(tracklets.keys())
    
    # Limit to max_tracklets if needed
    if len(track_ids) > max_tracklets:
        print(f"Limiting visualization to {max_tracklets} tracklets (out of {len(track_ids)})")
        track_ids = sorted(track_ids)[:max_tracklets]
    
    # Extract mean features for each tracklet
    features = []
    valid_track_ids = []
    
    print(f"Extracting features for {len(track_ids)} tracklets...")
    for track_id in track_ids:
        mean_feature = tracklets[track_id]._mean_
        
        # Skip tracklets with no valid features
        if mean_feature is None or (hasattr(mean_feature, 'size') and mean_feature.size == 0):
            print(f"Warning: No valid features for track ID {track_id}")
            continue
            
        # Convert features to numpy arrays if needed
        if hasattr(mean_feature, 'cpu'):
            # Convert PyTorch tensors to numpy
            mean_feature = mean_feature.cpu().detach().numpy()
            
        # Flatten if needed
        if len(mean_feature.shape) > 1:
            mean_feature = mean_feature.flatten()
            
        # Add to our lists
        features.append(mean_feature)
        valid_track_ids.append(track_id)
    
    # Check if we have enough features
    if len(features) < 2:
        print("Error: Not enough valid features to create a distance matrix")
        return None, valid_track_ids
    
    # Convert features to numpy array
    features_array = np.array(features)
    
    # Compute distance matrix
    print(f"Computing {distance_metric} distances between {len(features)} tracklets...")
    distance_matrix = cdist(features_array, features_array, metric=distance_metric)
    
    # Auto-determine figure size if not provided
    if figsize is None:
        # Base size plus additional space for each tracklet
        base_size = 8
        size_per_tracklet = 0.2
        size = base_size + len(valid_track_ids) * size_per_tracklet
        figsize = (size, size)
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Determine annotation parameters
    if show_values:
        # Use smaller font as number of tracklets increases
        if len(valid_track_ids) > 30:
            fontsize = 'xx-small'
        elif len(valid_track_ids) > 20:
            fontsize = 'x-small'
        else:
            fontsize = 'small'
        
        # Plot heatmap with annotations
        ax = sns.heatmap(
            distance_matrix, 
            cmap=cmap,
            xticklabels=valid_track_ids,
            yticklabels=valid_track_ids,
            vmin=0, 
            vmax=1 if distance_metric == 'cosine' else None,
            annot=True,
            fmt=".2f",
            annot_kws={"size": fontsize}
        )
    else:
        # Plot heatmap without annotations
        ax = sns.heatmap(
            distance_matrix, 
            cmap=cmap,
            xticklabels=valid_track_ids,
            yticklabels=valid_track_ids,
            vmin=0, 
            vmax=1 if distance_metric == 'cosine' else None
        )
    
    # Add title and labels
    plt.title(f'{distance_metric.capitalize()} Distance Matrix Between {len(valid_track_ids)} Tracklets')
    plt.xlabel('Track ID')
    plt.ylabel('Track ID')
    
    # Rotate x labels for better readability
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    
    # Add colorbar title
    cbar = ax.collections[0].colorbar
    cbar.set_label(f'{distance_metric.capitalize()} Distance (lower = more similar)')
    
    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Save visualization
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Distance matrix visualization saved to: {output_path}")

    plt.close()
    
    return distance_matrix, valid_track_ids

def check_time_overlap(tracklet1, tracklet2):
    """Check if two tracklets overlap in time"""
    frames1 = set(tracklet1.frames)
    frames2 = set(tracklet2.frames)
    return len(frames1.intersection(frames2)) > 0

def calculate_err(first_dist, second_dist):
    """
    Calculate error/difference between first and second best matches.
    This helps determine if the match is confident or uncertain.
    """
    if second_dist == 0:
        return 100  # Very confident if no second candidate
    
    # Calculate ratio between first and second distances
    return 100 * (1 - (first_dist / second_dist))

def define_reliable_uncertain_zones(frame_shape):
    """
    Define reliable and uncertain zones for people tracking.
    
    Args:
        frame_shape: Frame dimensions as (height, width)
        
    Returns:
        reliable_zone: Polygon coordinates for reliable zone
        uncertain_zones: List of polygons for uncertain zones
    """
    height, width = frame_shape
        
    # Define margins for reliable zone
    margin_x = int(width * 0.1)  # 10% margin from sides
    margin_y_near = int(height * 0.1)  # 10% margin from bottom
    margin_y_far = int(height * 0.2)   # 20% margin from top
    
    # Define reliable zone (central area)
    reliable_zone = np.array([
        [margin_x, margin_y_near],
        [width - margin_x, margin_y_near],
        [width - margin_x, height - margin_y_far],
        [margin_x, height - margin_y_far]
    ], dtype=np.int32)

    # Define uncertain zones (edges)
    left_edge = np.array([
        [0, 0], [margin_x, 0], [margin_x, height], [0, height]
    ], dtype=np.int32)
    
    right_edge = np.array([
        [width - margin_x, 0], [width, 0], [width, height], [width - margin_x, height]
    ], dtype=np.int32)
    
    near_edge = np.array([
        [margin_x, 0], [width - margin_x, 0], [width - margin_x, margin_y_near], [margin_x, margin_y_near]
    ], dtype=np.int32)
    
    far_edge = np.array([
        [margin_x, height - margin_y_far], [width - margin_x, height - margin_y_far],
        [width - margin_x, height], [margin_x, height]
    ], dtype=np.int32)
    
    uncertain_zones = [left_edge, right_edge, near_edge, far_edge]
    return reliable_zone, uncertain_zones

def filter_tracklets_by_zone(tracklets, reliable_zone, uncertain_zones, area_threshold_percentage=0.01):
    """
    Filter tracklets into reliable and uncertain groups based on zone and size.
    """
    reliable_tracklets = []
    uncertain_tracklets = []
    
    # Full frame area (1920 x 1080)
    full_frame_area = 1920 * 1080
    area_threshold = area_threshold_percentage * full_frame_area
    
    for tracklet in tracklets:
        # Calculate average area of bounding boxes
        total_area = sum((bbox[2]-bbox[0]) * (bbox[3]-bbox[1]) for bbox in tracklet.bboxes)
        avg_area = total_area / len(tracklet.bboxes) if tracklet.bboxes else 0
        
        # Check if the area is large enough
        size_reliable = avg_area >= area_threshold
        
        # Set outzone polygons and refine
        tracklet.outPolygonList = uncertain_zones
        tracklet.refine_tracklets()
        
        # Check if tracklet is in reliable zones
        zone_reliable = (len(tracklet.frames) > 0 and 
                        len(tracklet.frames) >= 0.5 * len(tracklet.backup_frames))
        
        # Skip tracklets that are too short in duration
        if not tracklet.frames or max(tracklet.frames) - min(tracklet.frames) < 60:
            tracklet.restore()
            continue
        
        # A tracklet is reliable if it's in reliable zone AND has sufficient size and duration
        if (zone_reliable and size_reliable and 
            max(tracklet.frames) - min(tracklet.frames) > 60):
            tracklet.is_reliable = True
            tracklet.restore()
            reliable_tracklets.append(tracklet)
        else:
            tracklet.is_reliable = False
            tracklet.restore()
            uncertain_tracklets.append(tracklet)
    
    return reliable_tracklets, uncertain_tracklets
def find_anchor_frame(tracklets, top_n=5):
    """
    Find the optimal anchor frame based on:
    1. PRIORITY: Frame with maximum number of people
    2. Among frames with max people: lowest similarity between people
    
    Args:
        tracklets: List of tracklet objects
        top_n: Number of candidate frames to consider
        
    Returns:
        Optimal frame number
    """
    # Step 1: Count people per frame and track which tracklets are in each frame
    frame_counts = defaultdict(int)
    frame_tracklets = defaultdict(list)
    
    for tracklet in tracklets:
        for frame in tracklet.frames:
            frame_counts[frame] += 1
            frame_tracklets[frame].append(tracklet)
    
    if not frame_counts:
        return 0
    
    # Step 2: Group frames by people count
    people_count_to_frames = defaultdict(list)
    for frame, count in frame_counts.items():
        people_count_to_frames[count].append(frame)
    
    # Step 3: Get the maximum people count
    max_people = max(people_count_to_frames.keys())
    max_people_frames = people_count_to_frames[max_people]
    logging.info(f"Found {len(max_people_frames)} frames with maximum people count: {max_people}")
    
    # Step 4: Find the frame with the smallest edge sum among max_people_frames
    best_frame = None
    smallest_edge_sum = float('inf')
    
    for frame in max_people_frames:
        frame_track_list = frame_tracklets[frame]
        
        # Calculate similarity matrix (sum of edges)
        edge_sum = 0
        n_comparisons = 0
        
        for i in range(len(frame_track_list)):
            for j in range(i+1, len(frame_track_list)):
                # Calculate similarity (1 - cosine distance)
                feat_i = frame_track_list[i]._mean_
                feat_j = frame_track_list[j]._mean_
                
                if feat_i is not None and feat_j is not None:
                    similarity = 1.0 - distance.cosine(feat_i, feat_j)
                    edge_sum += similarity
                    n_comparisons += 1
        
        # Normalize by number of comparisons
        if n_comparisons > 0:
            normalized_edge_sum = edge_sum / n_comparisons
            logging.info(f"Frame {frame}: {max_people} people, edge sum = {normalized_edge_sum:.4f}")
            
            if normalized_edge_sum < smallest_edge_sum:
                smallest_edge_sum = normalized_edge_sum
                best_frame = frame
    
    if best_frame is None:
        best_frame = max_people_frames[0]
        logging.info(f"No valid frame found, using first frame with max people: {best_frame}")
    else:
        logging.info(f"Selected optimal anchor frame {best_frame} with {max_people} people")
    
    return best_frame

def initialize_clusters_from_anchor(tracklets, anchor_frame):
    """
    Initialize clusters from anchor frame and tracklets with same parent_id
    
    Args:
        tracklets: List of tracklet objects
        anchor_frame: The selected anchor frame
        
    Returns:
        Dictionary of initial clusters and set of used tracklet IDs
    """
    anchor_clusters = {}
    cluster_id = 0
    initial_tracklet_ids = set()
    
    # Step 1: Create clusters for tracklets in the anchor frame
    for tracklet in tracklets:
        if anchor_frame in tracklet.frames:
            logging.info(f"Tracklet {tracklet.track_id} used as initial cluster from anchor frame")
            anchor_clusters[cluster_id] = [tracklet]
            initial_tracklet_ids.add(tracklet.track_id)
            cluster_id += 1
    
    # Step 2: Track parent_ids used by initial tracklets
    parent_ids_used = set()
    for tracklet in tracklets:
        if tracklet.track_id in initial_tracklet_ids:
            if hasattr(tracklet, 'parent_id') and tracklet.parent_id is not None:
                parent_ids_used.add(tracklet.parent_id)
    
    # Step 3: Add tracklets with same parent_id as separate clusters
    for tracklet in tracklets:
        if tracklet.track_id not in initial_tracklet_ids:
            # Check if tracklet shares parent_id with any initial tracklet
            if (hasattr(tracklet, 'parent_id') and 
                tracklet.parent_id is not None and 
                tracklet.parent_id in parent_ids_used):
                
                logging.info(f"Tracklet {tracklet.track_id} used as initial cluster (shares parent_id)")
                anchor_clusters[cluster_id] = [tracklet]
                initial_tracklet_ids.add(tracklet.track_id)
                cluster_id += 1
    for tracklet in tracklets:
        if tracklet.track_id==14:
            anchor_clusters[cluster_id]=[tracklet]
            initial_tracklet_ids.add(tracklet.track_id)
            cluster_id+=1
    
    logging.info(f"Created {len(anchor_clusters)} initial clusters from anchor frame {anchor_frame}")
    return anchor_clusters, initial_tracklet_ids

def split_tracklets_by_length(tracklets, clustered_ids, min_length_threshold=100):
    """
    Split remaining tracklets into long and short based on temporal length
    
    Args:
        tracklets: List of all tracklets
        clustered_ids: Set of already clustered tracklet IDs
        min_length_threshold: Minimum frames to be considered 'long'
        
    Returns:
        Tuple of (long_tracklets, short_tracklets)
    """
    remaining_tracklets = [t for t in tracklets if t.track_id not in clustered_ids]
    
    long_tracklets = []
    short_tracklets = []
    
    for tracklet in remaining_tracklets:
        temporal_length = max(tracklet.frames) - min(tracklet.frames) if tracklet.frames else 0
        
        if temporal_length >= min_length_threshold:
            long_tracklets.append(tracklet)
        else:
            short_tracklets.append(tracklet)
    
    logging.info(f"Split remaining tracklets: {len(long_tracklets)} long, {len(short_tracklets)} short")
    return long_tracklets, short_tracklets

def is_compatible_with_cluster(tracklet, cluster_tracklets):
    """
    Check if tracklet can be added to cluster without creating conflicts
    """
    for existing_tracklet in cluster_tracklets:
        # Check for time overlap
        if check_time_overlap(tracklet, existing_tracklet):
            return False
            
        # Check for parent relationship (split tracklets from same parent cannot be in same cluster)
        if (hasattr(tracklet, 'parent_id') and hasattr(existing_tracklet, 'parent_id') and
            tracklet.parent_id is not None and tracklet.parent_id == existing_tracklet.parent_id):
            return False
    
    return True

def calculate_cluster_feature(cluster_tracklets):
    """
    Calculate cluster feature based on equation (8): mean of tracklet features
    
    Args:
        cluster_tracklets: List of tracklets in the cluster
        
    Returns:
        Mean feature vector of the cluster
    """
    cluster_features = []
   
    for tracklet in cluster_tracklets:
        tracklet_feature = tracklet._mean_
        if tracklet_feature is not None:
            cluster_features.append(tracklet_feature)
    
    if cluster_features:
        return np.mean(cluster_features, axis=0)
    else:
        return None


def r_matching_single_camera(tracklets, frame_shape, distance_matrix=None, valid_track_ids=None, similarity_threshold=0.2, err_threshold=30):
    """
    Main R-matching algorithm for single camera tracking
    
    Steps:
    1. Define reliable/uncertain zones and filter tracklets
    2. Find anchor frame and initialize clusters
    3. Split remaining tracklets into long and short
    4. Match long tracklets to clusters using iterative assignment
    5. Apply R-matching for short tracklets and uncertain tracklets
    
    Args:
        tracklets: List of tracklet objects
        frame_shape: Tuple of (height, width)
        distance_matrix: Optional pre-computed distance matrix
        valid_track_ids: Optional list of tracklet IDs corresponding to distance_matrix indices
        similarity_threshold: Threshold for cosine distance
        err_threshold: Threshold for confident matches
        
    Returns:
        Dictionary mapping track IDs to cluster IDs
    """
    logging.info("Starting R-matching single camera algorithm")
    
    # Step 1: Define reliable and uncertain zones
    reliable_zone, uncertain_zones = define_reliable_uncertain_zones(frame_shape)
    reliable_tracklets, uncertain_tracklets = filter_tracklets_by_zone(
        tracklets, reliable_zone, uncertain_zones)
    
    logging.info(f"Filtered tracklets: {len(reliable_tracklets)} reliable, {len(uncertain_tracklets)} uncertain")
    
    if not reliable_tracklets:
        logging.warning("No reliable tracklets found, using all tracklets")
        reliable_tracklets = tracklets
    
    # Step 2: Find anchor frame and initialize clusters
    anchor_frame = find_anchor_frame(reliable_tracklets)
    initial_clusters, clustered_ids = initialize_clusters_from_anchor(reliable_tracklets, anchor_frame)
    
    # Step 3: Split remaining tracklets into long and short
    long_tracklets, short_tracklets = split_tracklets_by_length(
        reliable_tracklets, clustered_ids, min_length_threshold=100)
    
    # Step 4: Match long tracklets to clusters
    logging.info(f"Matching {len(long_tracklets)} long tracklets to clusters")
    updated_clusters, remaining_unmatched = match_long_tracklets_to_clusters(
        long_tracklets, initial_clusters, distance_matrix, valid_track_ids, 
        similarity_threshold, err_threshold)
    
    # Step 5: Match remaining unmatched tracklets and short tracklets
    logging.info(f"Matching {len(remaining_unmatched)} unmatched and {len(short_tracklets)} short tracklets")
    final_clusters = match_remain(
        remaining_unmatched, short_tracklets, updated_clusters, 
        distance_matrix, valid_track_ids, similarity_threshold=0.25)
    
    # Step 6: Handle uncertain tracklets separately
    if uncertain_tracklets:
        logging.info(f"Processing {len(uncertain_tracklets)} uncertain tracklets")
        final_clusters = match_remain(
            uncertain_tracklets, [], final_clusters, distance_matrix, valid_track_ids, similarity_threshold=0.25)
    
    # Create mapping from track_id to cluster_id
    track_to_cluster = {}
    for cluster_id, cluster_tracklets in final_clusters.items():
        for tracklet in cluster_tracklets:
            track_to_cluster[tracklet.track_id] = cluster_id
    
    # Print summary
    logging.info("=" * 50)
    logging.info("FINAL CLUSTERING RESULTS:")
    logging.info("=" * 50)
    
    cluster_sizes = [(cluster_id, len(tracklets)) for cluster_id, tracklets in final_clusters.items()]
    cluster_sizes.sort(key=lambda x: x[1], reverse=True)
    
    for cluster_id, size in cluster_sizes:
        tracklet_ids = [str(t.track_id) for t in final_clusters[cluster_id]]
        logging.info(f"Cluster {cluster_id}: {size} tracklets - {', '.join(tracklet_ids)}")
    
    logging.info(f"Total clusters: {len(final_clusters)}")
    logging.info(f"Total tracklets processed: {len(tracklets)}")
    
    return track_to_cluster


def match_long_tracklets_to_clusters(long_tracklets, clusters, distance_matrix, valid_track_ids, similarity_threshold=0.25, err_threshold=30):
    """
    Match long tracklets to existing clusters using a pre-computed distance matrix.
    Add a tracklet directly if it has a distinctive match (either only one valid cluster or 
    very high confidence). Update the distance matrix after each addition.
    
    Args:
        long_tracklets: List of long tracklets to be matched
        clusters: Dictionary of existing clusters
        distance_matrix: Pre-computed distance matrix between tracklets
        valid_track_ids: List of tracklet IDs corresponding to distance_matrix indices
        similarity_threshold: Maximum distance for valid match
        err_threshold: Minimum confidence threshold for assignment
        
    Returns:
        Tuple of (updated clusters dictionary, list of remaining unmatched tracklets)
    """
    remaining_tracklets = long_tracklets.copy()
    unmatched_tracklets = []
    
    # Create a mapping from track_id to index in the distance matrix
    track_id_to_index = {track_id: i for i, track_id in enumerate(valid_track_ids)}
    
    # Create a mapping from track_id to cluster_id
    track_to_cluster = {}
    for cluster_id, cluster_tracklets in clusters.items():
        for tracklet in cluster_tracklets:
            track_to_cluster[tracklet.track_id] = cluster_id
    
    # Process each tracklet
    made_progress = True
    while made_progress and remaining_tracklets:
        made_progress = False
        
        # Make a copy to safely modify during iteration
        for tracklet in list(remaining_tracklets):
            # Skip if tracklet ID is not in the distance matrix
            if tracklet.track_id not in track_id_to_index:
                remaining_tracklets.remove(tracklet)
                unmatched_tracklets.append(tracklet)
                continue
                
            tracklet_pairs = []
            tracklet_idx = track_id_to_index[tracklet.track_id]
            
            # For each cluster, compute average distance to this tracklet
            for cluster_id, cluster_tracklets in clusters.items():
                # Skip specific clusters if needed
                if cluster_id == 6:
                    continue
                
                # Check compatibility (e.g., time overlap)
                if not is_compatible_with_cluster(tracklet, cluster_tracklets):
                    continue
                
                # Calculate average distance to cluster using distance matrix
                cluster_distances = []
                for cluster_tracklet in cluster_tracklets:
                    if cluster_tracklet.track_id in track_id_to_index:
                        cluster_idx = track_id_to_index[cluster_tracklet.track_id]
                        cluster_distances.append(distance_matrix[tracklet_idx][cluster_idx])
                
                if not cluster_distances:
                    continue
                
                # Average distance to this cluster
                avg_dist = min(cluster_distances)
                
                if avg_dist <= similarity_threshold :
                    tracklet_pairs.append((avg_dist, cluster_id))
            
            # Sort by distance (ascending)
            tracklet_pairs.sort(key=lambda x: x[0])
            
            # No viable clusters for this tracklet
            if len(tracklet_pairs) == 0:
                continue
            
            # Check if this tracklet has a distinctive match
            is_distinctive = False
            best_cluster_id = None
            best_dist = None
            
            if len(tracklet_pairs) == 1:
                # Only one possible match - very distinctive
                is_distinctive = True
                best_dist, best_cluster_id = tracklet_pairs[0]
            elif len(tracklet_pairs) >= 2:
                # Check if first match is much better than second
                best_dist, best_cluster_id = tracklet_pairs[0]
                second_dist, _ = tracklet_pairs[1]
                
                error = calculate_err(best_dist, second_dist)
                if error >= err_threshold:
                    is_distinctive = True
            
            # If distinctive match found, add to cluster and update distance matrix
            if is_distinctive:
                clusters[best_cluster_id].append(tracklet)
                remaining_tracklets.remove(tracklet)
                
                # Update track_to_cluster mapping
                track_to_cluster[tracklet.track_id] = best_cluster_id
                
                # Update distance matrix (set distance to infinity between this tracklet and all tracklets in other clusters)
                tracklet_idx = track_id_to_index[tracklet.track_id]
                for track_id, cluster_id in track_to_cluster.items():
                    if cluster_id != best_cluster_id and track_id in track_id_to_index:
                        idx = track_id_to_index[track_id]
                        # Set distance to infinity in both directions
                        distance_matrix[tracklet_idx][idx] = float('inf')
                        distance_matrix[idx][tracklet_idx] = float('inf')
                
                logging.info(f"Added tracklet {tracklet.track_id} to cluster {best_cluster_id} with distance {best_dist:.4f}")
                made_progress = True
        
        # If no progress in this iteration, exit loop
        if not made_progress:
            break
    
    # All remaining tracklets are unmatched
    unmatched_tracklets.extend(remaining_tracklets)
    
    return clusters, unmatched_tracklets


def match_remain(unmatched_tracklets, short_tracklets, clusters, distance_matrix, valid_track_ids, similarity_threshold=0.25):
    """
    Match remaining unmatched tracklets and short tracklets to clusters using distance matrix directly
    
    Args:
        unmatched_tracklets: List of unmatched tracklets from long tracklet matching
        short_tracklets: List of short tracklets to be matched
        clusters: Dictionary of existing clusters
        distance_matrix: Pre-computed distance matrix between tracklets
        valid_track_ids: List of tracklet IDs corresponding to distance_matrix indices
        similarity_threshold: Maximum distance for valid match
        
    Returns:
        Updated clusters dictionary
    """
    # Combine all tracklets to be matched
    tracklets_to_match = unmatched_tracklets + short_tracklets
    
    if not tracklets_to_match:
        return clusters
    
    logging.info(f"Matching {len(tracklets_to_match)} tracklets using distance matrix")
    
    # Create a mapping from track_id to index in the distance matrix
    track_id_to_index = {track_id: i for i, track_id in enumerate(valid_track_ids)}
    
    # Filter tracklets that are in the distance matrix
    valid_tracklets = []
    invalid_tracklets = []
    for tracklet in tracklets_to_match:
        if tracklet.track_id in track_id_to_index:
            valid_tracklets.append(tracklet)
        else:
            invalid_tracklets.append(tracklet)
    
    # Prepare query indices and features
    query_indices = []
    query_tracklet_ids = []
    
    for tracklet in valid_tracklets:
        query_indices.append(track_id_to_index[tracklet.track_id])
        query_tracklet_ids.append(tracklet.track_id)
    
    # Prepare gallery indices and features
    gallery_indices = []
    gallery_cluster_ids = []
    
    # Create a mapping of cluster representatives
    cluster_representatives = {}
    
    for cluster_id, cluster_tracklets in clusters.items():
        if cluster_id == 6:  # Skip specific clusters if needed
            continue
            
        # Find representative tracklets for this cluster
        representatives = []
        for tracklet in cluster_tracklets:
            if tracklet.track_id in track_id_to_index:
                representatives.append(track_id_to_index[tracklet.track_id])
        
        if representatives:
            cluster_representatives[cluster_id] = representatives
            gallery_cluster_ids.append(cluster_id)
            # Use all representatives
            for rep_idx in representatives:
                gallery_indices.append((rep_idx, cluster_id))
    
    if not query_indices or not gallery_indices:
        # No valid queries or galleries, create new clusters for all tracklets
        next_cluster_id = max(clusters.keys()) + 1 if clusters else 0
        for tracklet in tracklets_to_match:
            clusters[next_cluster_id] = [tracklet]
            logging.info(f"Created new cluster {next_cluster_id} for tracklet {tracklet.track_id} (no valid matches)")
            next_cluster_id += 1
        return clusters
    
    # Extract distance sub-matrix for matching
    query_gallery_matrix = np.zeros((len(query_indices), len(gallery_indices)))
    for i, q_idx in enumerate(query_indices):
        for j, (g_idx, _) in enumerate(gallery_indices):
            query_gallery_matrix[i][j] = distance_matrix[q_idx][g_idx]
    
    # We'll use the pre-computed distance matrix directly
    reranked_matrix = query_gallery_matrix
    

    matched_tracklets = []
    
    for i, tracklet_id in enumerate(query_tracklet_ids):
        tracklet = next(t for t in valid_tracklets if t.track_id == tracklet_id)
        
        # Find compatible clusters
        compatible_indices = []
        for j, (_, cluster_id) in enumerate(gallery_indices):
            if is_compatible_with_cluster(tracklet, clusters[cluster_id]):
                compatible_indices.append(j)
        
        if not compatible_indices:
            continue
        
        # Find best match among compatible clusters
        best_j = min(compatible_indices, key=lambda j: reranked_matrix[i][j])
        best_dist = reranked_matrix[i][best_j]
        best_cluster_id = gallery_indices[best_j][1]
        
        # Only assign if distance is below threshold
        if best_dist <= similarity_threshold:
            clusters[best_cluster_id].append(tracklet)
            matched_tracklets.append(tracklet)
            logging.info(f"Matrix matching: Assigned tracklet {tracklet_id} to cluster {best_cluster_id} "
                        f"with distance {best_dist:.4f}")
    
    # Create new clusters for unmatched tracklets
    unmatched = [t for t in valid_tracklets if t not in matched_tracklets] + invalid_tracklets
    if not unmatched:
        logging.info(f"All tracklets matched to existing clusters")
        return clusters
    
    logging.info(f"Clustering {len(unmatched)} unmatched tracklets")
    
    # Create distance matrix for unmatched tracklets
    unmatched_distance_matrix = np.ones((len(unmatched), len(unmatched)))
    # Fill in distance matrix for unmatched tracklets
    for i in range(len(unmatched)):
        for j in range(len(unmatched)):
            if i == j:
                unmatched_distance_matrix[i][j] = 0.0
                continue
                
            # Check if tracklets overlap in time (cannot be same person)
            if check_time_overlap(unmatched[i], unmatched[j]):
                unmatched_distance_matrix[i][j] = 1.0
                continue
                
            # Use precomputed distance matrix if available
            if (unmatched[i].track_id in track_id_to_index and 
                unmatched[j].track_id in track_id_to_index):
                i_idx = track_id_to_index[unmatched[i].track_id]
                j_idx = track_id_to_index[unmatched[j].track_id]
                unmatched_distance_matrix[i][j] = distance_matrix[i_idx][j_idx]
            else:
                # Fallback to a default high distance
                unmatched_distance_matrix[i][j] = 0.8
    next_cluster_id = max(clusters.keys()) + 1 if clusters else 0
    
    # Group tracklets based on distance
    cluster_assignments = {}
    remaining_tracklets = set(range(len(unmatched)))
    
    # Process tracklets in order of confidence/reliability (longer tracklets first)
    tracklet_indices = sorted(range(len(unmatched)), 
                             key=lambda i: len(unmatched[i].frames), 
                             reverse=True)
    
    
    
    for idx in tracklet_indices:
        if idx not in remaining_tracklets:
            continue
            
        # Start a new cluster with this tracklet
        cluster_tracklets = [idx]
        remaining_tracklets.remove(idx)
        
        # Find other tracklets that are close to this one and compatible
        for other_idx in list(remaining_tracklets):
            # Skip if tracklets overlap in time
            if check_time_overlap(unmatched[idx], unmatched[other_idx]):
                continue
                
            # Check if tracklets are similar enough
            if unmatched_distance_matrix[idx][other_idx] <= similarity_threshold:
                # Check compatibility with all tracklets already in cluster
                is_compatible = True
                for cluster_idx in cluster_tracklets:
                    if check_time_overlap(unmatched[cluster_idx], unmatched[other_idx]):
                        is_compatible = False
                        break
                
                if is_compatible:
                    cluster_tracklets.append(other_idx)
                    remaining_tracklets.remove(other_idx)
        
        # Create a new cluster with these tracklets
        tracklet_ids = [unmatched[i].track_id for i in cluster_tracklets]
        clusters[next_cluster_id] = [unmatched[i] for i in cluster_tracklets]
        
        logging.info(f"Created new cluster {next_cluster_id} with {len(cluster_tracklets)} tracklets: {tracklet_ids}")
        next_cluster_id += 1
    
    logging.info(f"Matrix matching matched {len(matched_tracklets)}/{len(tracklets_to_match)} tracklets")
    logging.info(f"Created {next_cluster_id - (max(clusters.keys()) + 1 if clusters else 0)} new clusters for unmatched tracklets")
    
    return clusters


def dist_based_re_ranking(original_dist, k1=20, k2=6, lambda_value=0.3):
    """
    Re-ranking implementation from Zhong et al. CVPR 2017 adapted for distance matrices
    
    Args:
        original_dist: original distance matrix with shape (m+n, m+n), 
                       where m is the number of query samples and n is the number of gallery samples
        k1: hyperparameter for k-reciprocal neighbors
        k2: hyperparameter for Jaccard distance expansion
        lambda_value: weighting parameter for final distance
        
    Returns:
        final_dist: re-ranked distance matrix
    """
    # Ensure the distance is between 0 and 2 (for cosine distance)
    original_dist = np.clip(original_dist, 0, 2)
    
    # For numerical stability
    original_dist = np.power(original_dist, 2)
    original_dist = np.transpose(1.0 * original_dist / np.max(original_dist, axis=0))
    V = np.zeros_like(original_dist).astype(np.float32)
    
    # Get dimensions
    m, n = original_dist.shape
    
    # Get k-reciprocal neighbors
    total_num = original_dist.shape[0]
    
    # k-reciprocal neighbors
    forward_k_neigh_index = np.argsort(original_dist, axis=1)[:, :k1 + 1]
    backward_k_neigh_index = np.argsort(original_dist.T, axis=1)[:, :k1 + 1]
    
    for i in range(total_num):
        k_reciprocal_index = np.intersect1d(forward_k_neigh_index[i, :], 
                                            backward_k_neigh_index[:, 1:].reshape(-1))
        k_reciprocal_expansion_index = k_reciprocal_index
        
        # k2-expansion
        for candidate in k_reciprocal_index:
            candidate_k_reciprocal_index = np.intersect1d(
                forward_k_neigh_index[candidate, :], 
                backward_k_neigh_index[:, 1:].reshape(-1))
            
            if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2 / 3 * len(
                    candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)
        
        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        
        # Local query expansion and calculate Jaccard distance
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = weight / np.sum(weight)
    
    # Compute Jaccard distance
    jaccard_dist = np.sum(np.abs(V - V.T), axis=1) / 2.0
    
    # If we're dealing with a query-gallery setup (m × n matrix)
    if m != n:
        jaccard_dist_qg = jaccard_dist[:m, m:]
        
        # Combine distances
        final_dist = jaccard_dist_qg * (1 - lambda_value) + original_dist[:m, m:] * lambda_value
    else:
        # For square matrix case
        final_dist = jaccard_dist * (1 - lambda_value) + original_dist * lambda_value
    
    return final_dist
def split_tracklets_from_log(tracklets, log_file="/home/aidev/workspace/reid/Thesis/reid-2024/trash/2025-05-22 20-42-37-2de4f2989e1f4e4298b81da615925c83/example.log"):
    """
    Split tracklets based on ID switches detected from log file
    
    Args:
        tracklets: Dictionary of original tracklets
        log_file: Path to log file containing ID switch information
        
    Returns:
        Dictionary of split tracklets with updated IDs
    """
    # Extract ID switches from log
    id_switches = extract_id_switches_from_log(log_file)
    filtered_switches = filter_nearby_switches(id_switches, frame_gap=20)
    
    # Create map of tracklet IDs to switch frames
    switch_frame_map = {}
    for entry in filtered_switches:
        tid = int(entry['track_id'])
        frame = int(entry['frame'])
        switch_frame_map[tid] = frame
    
    # Filter out switches too close to tracklet end
    valid_switches = {}
    for tid, switch_frame in switch_frame_map.items():
        tracklet = tracklets.get(tid)
        if tracklet and tracklet.frames:
            max_frame = max(tracklet.frames)
            if abs(max_frame - switch_frame) >= 100:  # At least 30 frames from end
                valid_switches[tid] = switch_frame
    
    logging.info(f"Found {len(valid_switches)} valid ID switches to process")
    
    # Split tracklets
    split_tracklets = {}
    next_id = max(tracklets.keys()) + 1 if tracklets else 0
    
    for track_id, tracklet in tracklets.items():
        if track_id in valid_switches:
            switch_frame = valid_switches[track_id]
            logging.info(f"Splitting tracklet {track_id} at frame {switch_frame}")
            
            # Create two new tracklets
            track_a = Tracklet(track_id, tracklet.camera_id)  # Keep original ID
            track_b = Tracklet(next_id, tracklet.camera_id)   # New ID
            
            # Split detections
            for i, frame in enumerate(tracklet.frames):
                if frame < switch_frame:
                    track_a.add_detection(
                        tracklet.frames[i], tracklet.bboxes[i], 
                        tracklet.features[i], tracklet.scores[i])
                else:
                    track_b.add_detection(
                        tracklet.frames[i], tracklet.bboxes[i], 
                        tracklet.features[i], tracklet.scores[i])
            
            # Set parent ID for tracking relationship
            track_a.parent_id = track_id
            track_b.parent_id = track_id
            
            # Only keep splits with sufficient frames
            if len(track_a.frames) >= 100:
                split_tracklets[track_id] = track_a
                logging.info(f"  Part A: {len(track_a.frames)} frames")
            else:
                logging.warning(f"  Part A too short ({len(track_a.frames)} frames), keeping original")
                split_tracklets[track_id] = tracklet
                continue
                
            if len(track_b.frames) >= 100:
                split_tracklets[next_id] = track_b
                logging.info(f"  Part B (ID {next_id}): {len(track_b.frames)} frames")
                next_id += 1
            else:
                logging.warning(f"  Part B too short ({len(track_b.frames)} frames), discarding")
        else:
            # No split needed
            split_tracklets[track_id] = tracklet
    
    logging.info(f"Tracklet splitting: {len(tracklets)} -> {len(split_tracklets)} tracklets")
    return split_tracklets

def detect_additional_id_switches(tracklets, similarity_threshold=0.4):
    """
    Detect additional ID switches using GMM-based approach
    
    Args:
        tracklets: List or dict of tracklets
        similarity_threshold: Threshold for GMM detection
        
    Returns:
        List of corrected tracklets with additional splits
    """
    if isinstance(tracklets, dict):
        tracklets = list(tracklets.values())
    
    corrected_tracklets = []
    split_count = 0
    
    for tracklet in tqdm(tracklets, desc="Detecting additional ID switches"):
        if not tracklet.features or len(tracklet.features) < 15:
            corrected_tracklets.append(tracklet)
            continue
        
        # Detect ID switch using GMM
        has_switch, labels = tracklet.detect_id_switch(similarity_threshold)
        
        if has_switch:
            # Split tracklet
            track_a, track_b = tracklet.split_track(labels)
            
            if (track_a and track_b and 
                len(track_a.features) >= 10 and len(track_b.features) >= 10):
                
                # Set parent relationship
                track_a.parent_id = tracklet.track_id
                track_b.parent_id = tracklet.track_id
                
                corrected_tracklets.extend([track_a, track_b])
                split_count += 1
                logging.info(f"GMM split tracklet {tracklet.track_id} into {track_a.track_id} "
                           f"({len(track_a.frames)} frames) and {track_b.track_id} ({len(track_b.frames)} frames)")
            else:
                corrected_tracklets.append(tracklet)
        else:
            corrected_tracklets.append(tracklet)
    
    logging.info(f"GMM detection: split {split_count} additional tracklets")
    return corrected_tracklets
def save_tracklet_features(tracklets, output_dir):
    """
    Save individual tracklet features for multi-camera matching
    
    Args:
        tracklets: List of tracklet objects
        output_dir: Directory to save features
        
    Returns:
        Path to the saved features file
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract features from tracklets
    tracklet_features = {}
    for tracklet in tracklets:
        if hasattr(tracklet, '_mean_') and tracklet._mean_ is not None:
            feature = tracklet._mean_
            # Convert to numpy if needed
            if hasattr(feature, 'cpu'):
                feature = feature.cpu().detach().numpy()
            tracklet_features[str(tracklet.track_id)] = feature
    
    # Save features to file
    features_file = os.path.join(output_dir, "tracklet_features.pkl")
    with open(features_file, 'wb') as f:
        pickle.dump(tracklet_features, f)
    
    logging.info(f"Saved features for {len(tracklet_features)} tracklets to {features_file}")
    return features_file
def save_cluster_data(output_dir, corrected_tracklets, track_to_cluster):
    """
    Save cluster data for multi-camera matching including:
    1. Cluster assignments (which tracklets belong to which clusters)
    2. Cluster features (mean feature for each cluster)
    3. Track to cluster mapping
    
    Args:
        output_dir: Directory to save results
        corrected_tracklets: List of tracklet objects
        track_to_cluster: Dictionary mapping track IDs to cluster IDs
        
    Returns:
        Tuple of (clusters_json_file, cluster_features_file, mapping_file)
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Group tracklets by cluster
    clusters = {}
    for tracklet in corrected_tracklets:
        cluster_id = track_to_cluster.get(tracklet.track_id, -1)
        if cluster_id == -1:  # Skip tracklets not assigned to any cluster
            continue
            
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(tracklet)
    
    # 1. Save cluster assignments as JSON
    clusters_json = {}
    for cluster_id, tracklets_in_cluster in clusters.items():
        clusters_json[str(cluster_id)] = [int(t.track_id) for t in tracklets_in_cluster]
    
    clusters_json_file = os.path.join(output_dir, "clusters.json")
    with open(clusters_json_file, 'w') as f:
        json.dump(clusters_json, f, indent=2)
    logging.info(f"Saved cluster assignments to {clusters_json_file}")
    
    # 2. Calculate and save cluster features
    cluster_features = {}
    for cluster_id, tracklets_in_cluster in clusters.items():
        # Extract features from tracklets in this cluster
        features = []
        for tracklet in tracklets_in_cluster:
            if hasattr(tracklet, '_mean_') and tracklet._mean_ is not None:
                feature = tracklet._mean_
                # Convert to numpy if needed
                if hasattr(feature, 'cpu'):
                    feature = feature.cpu().detach().numpy()
                features.append(feature)
        
        # Calculate mean feature if we have any valid features
        if features:
            try:
                # Ensure features are numpy arrays with the same shape
                features = [f for f in features if isinstance(f, np.ndarray)]
                if features:
                    cluster_features[str(cluster_id)] = np.mean(features, axis=0)
            except Exception as e:
                logging.warning(f"Error calculating mean feature for cluster {cluster_id}: {e}")
    
    cluster_features_file = os.path.join(output_dir, "cluster_features.pkl")
    with open(cluster_features_file, 'wb') as f:
        pickle.dump(cluster_features, f)
    logging.info(f"Saved features for {len(cluster_features)} clusters to {cluster_features_file}")
    
    # 3. Save track to cluster mapping
    mapping = {str(t.track_id): track_to_cluster[t.track_id] for t in corrected_tracklets if t.track_id in track_to_cluster}
    mapping_file = os.path.join(output_dir, "track_to_cluster.json")
    with open(mapping_file, 'w') as f:
        json.dump(mapping, f, indent=2)
    logging.info(f"Saved track-to-cluster mapping to {mapping_file}")
    
    return clusters_json_file, cluster_features_file, mapping_file


def save_frame_ranges(tracklets, output_dir):
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract frame ranges
    frame_ranges = {}
    for tracklet in tracklets:
        if tracklet.frames:
            min_frame = min(tracklet.frames)
            max_frame = max(tracklet.frames)
            frame_ranges[str(tracklet.track_id)] = (int(min_frame), int(max_frame))
    # Save to file
    ranges_file = os.path.join(output_dir, "frame_ranges.json")
    with open(ranges_file, 'w') as f:
        json.dump(frame_ranges, f, indent=2)
    
    logging.info(f"Saved frame ranges for {len(frame_ranges)} tracklets to {ranges_file}")
    return ranges_file
def save_all_matching_data(output_dir, corrected_tracklets, track_to_cluster, camera_id=None):
    """
    Save all data needed for multi-camera matching in one function
    
    Args:
        output_dir: Directory to save results
        corrected_tracklets: List of tracklet objects
        track_to_cluster: Dictionary mapping track IDs to cluster IDs
        camera_id: Optional camera identifier to include in output files
        
    Returns:
        Dictionary with paths to all saved files
    """
    # Create camera-specific prefix if provided
    prefix = f"{camera_id}_" if camera_id else ""
    
    # Ensure output directory exists
    output_dir = os.path.join(output_dir, "matching_data")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Save cluster data
    clusters_file, features_file, mapping_file = save_cluster_data(
        output_dir, corrected_tracklets, track_to_cluster)
    
    # 2. Save tracklet features
    tracklet_features_file = save_tracklet_features(corrected_tracklets, output_dir)
    
    # 3. Save frame ranges
    frame_ranges_file = save_frame_ranges(corrected_tracklets, output_dir)
    
    # 4. Save additional metadata
    metadata = {
        "camera_id": camera_id,
        "num_tracklets": len(corrected_tracklets),
        "num_clusters": len(set(track_to_cluster.values())),
        "timestamp": __import__('datetime').datetime.now().isoformat()
    }
    
    metadata_file = os.path.join(output_dir, f"{prefix}metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Return paths to all saved files
    return {
        "clusters": clusters_file,
        "cluster_features": features_file,
        "track_to_cluster": mapping_file,
        "tracklet_features": tracklet_features_file,
        "frame_ranges": frame_ranges_file,
        "metadata": metadata_file
    }

def save_results(tracklets, track_to_cluster, output_file):
    """
    Save the results in the required format
    
    Args:
        tracklets: List of tracklets
        track_to_cluster: Dict mapping track IDs to cluster IDs
        output_file: Output file path
    """
    with open(output_file, 'w') as f:
        for tracklet in tracklets:
            cluster_id = track_to_cluster.get(tracklet.track_id, -1)
            
            for i in range(len(tracklet.frames)):
                frame = tracklet.frames[i]
                bbox = tracklet.bboxes[i]
                
                # Format: camera_id, cluster_id, frame, x, y, w, h, -1, -1
                line = f"{tracklet.camera_id},{cluster_id},{frame},{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]},-1,-1\n"
                f.write(line)
    
    logging.info(f"Results saved to {output_file}")

def visualize_distance_matrix(tracklets, output_path=None, max_tracklets=50):
    """
    Visualize the distance matrix between tracklets
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from scipy.spatial.distance import cdist
        
        if isinstance(tracklets, dict):
            tracklets = list(tracklets.values())
        
        if len(tracklets) > max_tracklets:
            logging.info(f"Limiting visualization to {max_tracklets} tracklets")
            tracklets = tracklets[:max_tracklets]
        
        # Extract features
        features = []
        track_ids = []
        
        for t in tracklets:
            if t.features:
                mean_feat = t._mean_
                if mean_feat is not None:
                    features.append(mean_feat)
                    track_ids.append(t.track_id)
        
        if len(features) < 2:
            logging.warning("Not enough features for distance matrix visualization")
            return
        
        # Compute distance matrix
        distance_matrix = cdist(features, features, metric='cosine')
        
        # Create visualization
        plt.figure(figsize=(max(12, len(features)//3), max(10, len(features)//3)))
        sns.heatmap(
            distance_matrix, 
            cmap='viridis_r',
            xticklabels=track_ids,
            yticklabels=track_ids,
            vmin=0, 
            vmax=1,
            annot=len(features) < 20,  # Only show values for small matrices
            fmt=".2f"
        )
        
        plt.title(f'Cosine Distance Matrix Between {len(features)} Tracklets')
        plt.xlabel('Tracklet ID')
        plt.ylabel('Tracklet ID')
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        
        if output_path:
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logging.info(f"Distance matrix saved to {output_path}")
        else:
            plt.show()
            
    except ImportError as e:
        logging.warning(f"Could not create visualization: {e}")

def visualize_results(session_dir, tracking_file, output_video):
    """
    Create visualization video with tracking results
    """
    try:
        # Find video files
        video_files = []
        for root, _, files in os.walk(session_dir):
            for file in files:
                if file.endswith((".mp4", ".avi", ".MOV")):
                    video_files.append(os.path.join(root, file))
        
        if not video_files:
            logging.error(f"No video files found in {session_dir}")
            return
        
        video_files.sort()
        
        # Read tracking results
        tracking_data = {}
        with open(tracking_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 7:
                    continue
                    
                camera_id, cluster_id, frame, x, y, w, h = parts[:7]
                frame = int(frame)
                cluster_id = int(cluster_id)
                bbox = [float(x), float(y), float(w), float(h)]
                
                if frame not in tracking_data:
                    tracking_data[frame] = []
                    
                tracking_data[frame].append((cluster_id, bbox))
        
        # Create output video
        cap = cv2.VideoCapture(video_files[0])
        if not cap.isOpened():
            logging.error(f"Could not open video: {video_files[0]}")
            return
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
        cap.release()
        
        # Generate colors
        np.random.seed(42)
        colors = {}
        
        # Process videos
        frame_idx = 0
        for video_file in video_files:
            logging.info(f"Processing video: {video_file}")
            cap = cv2.VideoCapture(video_file)
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Draw tracking results
                if frame_idx in tracking_data:
                    for cluster_id, bbox in tracking_data[frame_idx]:
                        if cluster_id not in colors:
                            colors[cluster_id] = tuple(map(int, np.random.randint(0, 255, 3)))
                        
                        x, y, w, h = [int(v) for v in bbox]
                        
                        cv2.rectangle(frame, (x, y), (x+w, y+h), colors[cluster_id], 2)
                        cv2.putText(frame, f"ID: {cluster_id}", (x, y-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[cluster_id], 2)
                
                cv2.putText(frame, f"Frame: {frame_idx}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                out.write(frame)
                frame_idx += 1
            
            cap.release()
        
        out.release()
        logging.info(f"Visualization saved to {output_video}")
        
    except Exception as e:
        logging.error(f"Error creating visualization: {e}")
def extract_tracklet_mean_frames(tracking_file, video_path, output_dir, tracklets_dict, margin=10):
    """
    Extract the exact frame used for each tracklet's mean feature calculation.
    
    Args:
        tracking_file: Path to tracking results file
        video_path: Path to source video
        output_dir: Directory to save crops
        tracklets_dict: Dictionary of tracklets with selected frames
        margin: Margin to add around bounding box
    """
    import cv2
    import os
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Special ID mapping
    id_mapping = {
        28: 3,   # Track 28 maps to ID 3 in tracking file
        29: 12   # Track 29 maps to ID 12 in tracking file
    }
    
    # Load tracking results
    detections = {}
    with open(tracking_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 6:
                continue
                
            frame_id = int(parts[0])
            track_id = int(parts[1])
            x, y, w, h = map(float, parts[2:6])
            
            detections[(track_id, frame_id)] = (x, y, w, h)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video dimensions
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Process each tracklet
    saved_images = []
    
    for track_id, tracklet in tracklets_dict.items():
        # Get the selected frame for this tracklet
        if hasattr(tracklet, '_selected_frames'):
            selected_frames = tracklet._selected_frames
        else:
            _, selected_frames = tracklet.mean_features(list(tracklets_dict.values()))
        
        # Map the track ID to the ID used in tracking file if needed
        tracking_id = id_mapping.get(track_id, track_id)
        
        # Process each selected frame
        for frame_id in selected_frames:
            # Check if we have this detection
            if (tracking_id, frame_id) not in detections:
                print(f"Warning: No detection for track {track_id} (tracking ID {tracking_id}), frame {frame_id}")
                continue
            
            # Get detection
            x1, y1, x2, y2 = detections[(tracking_id, frame_id)]
            
            # Seek to frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = cap.read()
            
            if not ret:
                print(f"Warning: Could not read frame {frame_id}")
                continue
            
            # Apply margin and crop
            x1 = int(max(0,x1))
            y1 = int(max(0, y1))
            x2 = int(min(width, x2))
            y2 = int(min(height, y2))
            
            crop = frame[y1:y2, x1:x2]
            
            # Save crop using the original track_id (not the mapped one)
            output_path = os.path.join(output_dir, f"track_{track_id}_frame_{frame_id}.jpg")
            cv2.imwrite(output_path, crop)
            saved_images.append(output_path)
            
            if track_id in id_mapping:
                print(f"Saved track {track_id} (tracking ID {tracking_id}), frame {frame_id}")
            else:
                print(f"Saved track {track_id}, frame {frame_id}")
    
    # Release video
    cap.release()
    
    print(f"Saved {len(saved_images)} images to {output_dir}")
    return saved_images


def process_session(session_dir, video_path=None):
    """
    Main function to process a session directory
    
    Args:
        session_dir: Path to session directory
        video_path: Optional path to video file for frame dimensions
        
    Returns:
        Path to final output file
    """
    session_id = os.path.basename(session_dir)
    logging.info(f"Processing session: {session_id}")
    
    # Input files
    tracking_file = os.path.join(session_dir, "tracking_results.txt")
    feature_file = os.path.join(session_dir, "features", "all_features.jsonl")
    
    # Output directory
    output_dir = os.path.join(session_dir, "offline_results")
    os.makedirs(output_dir, exist_ok=True)
    
    final_output_file = os.path.join(output_dir, "final_results.txt")
    
    # Check input files
    if not os.path.exists(tracking_file):
        logging.error(f"Tracking file not found: {tracking_file}")
        return None
        
    if not os.path.exists(feature_file):
        logging.error(f"Feature file not found: {feature_file}")
        return None
    
    # Get frame dimensions
    frame_shape = (1080, 1920)  # Default
    if video_path and os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_shape = (height, width)
            cap.release()
            logging.info(f"Using frame dimensions: {frame_shape}")
    
    # Step 1: Load data
    logging.info("Loading tracking data and features...")
    start_time = time.time()
    tracklets = load_data(tracking_file, feature_file)
    logging.info(f"Data loading completed in {time.time() - start_time:.2f} seconds")
    
    # Step 2: Split tracklets based on log
    logging.info("Splitting tracklets based on detected ID switches...")
    start_time = time.time()
    split_tracklets = split_tracklets_from_log(tracklets)
    
    logging.info(f"Tracklet splitting completed in {time.time() - start_time:.2f} seconds")

    # Extract the exact frames used for mean feature calculation
    saved_images = extract_tracklet_mean_frames(
        tracking_file=tracking_file,
        video_path="/home/aidev/workspace/reid/Thesis/reid-2024/app/assets/videos/cam2.MOV",
        output_dir=os.path.join("/home/aidev/workspace/reid/Thesis/reid-2024/Screenshot", "mean_feature_frames_cam2"),
        tracklets_dict=split_tracklets,
        margin=0
    )
    feature_extractor = ReIDFeatureExtractor(
        config_file_path='/home/aidev/workspace/reid/Thesis/Training/FPB/configs/cuhk_detected.yaml',
        model_weights_path='/home/aidev/workspace/reid/Thesis/Training/FPB/log/cuhk_detected/model.pth.tar-120',
        num_classes=702
    )
    print(saved_images)
    for image_path in saved_images:
        # Extract the tracklet ID from the filename
        match = re.search(r'track_(\d+)_frame', image_path)
        if match:
            tracklet_id = int(match.group(1))
            
            # Read the image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Failed to read image at {image_path}")
                continue
            
            # Extract feature
            feature = feature_extractor.inference([image])
            
            # Assign the feature to the corresponding tracklet
            if hasattr(feature, 'cpu'):
                # Convert PyTorch tensors to numpy
                feature = feature.cpu().detach().numpy()
            if tracklet_id in split_tracklets:
                split_tracklets[tracklet_id]._mean_ = feature.flatten()
            else:
                print(f"Warning: Tracklet ID {tracklet_id} not found in tracklets")
    # Step 3: Detect additional ID switches using GMM
    logging.info("Detecting additional ID switches using GMM...")
    start_time = time.time()
    distance_matrix, valid_track_ids = create_tracklet_distance_matrix(
    tracklets=split_tracklets,
    output_path="tracklet_distance_matrix.png",
    distance_metric="cosine",  # Change to "euclidean" if preferred
    show_values=True,          # Show actual distance values in cells
    max_tracklets=40,          # Limit for readability
    cmap="viridis_r"           # Other good options: "coolwarm", "RdBu_r", "YlGnBu"
)   
    
    
    logging.info(f"GMM ID switch detection completed in {time.time() - start_time:.2f} seconds")
    corrected_tracklets=list(split_tracklets.values())
    # Step 4: Apply R-matching algorithm
    logging.info("Applying R-matching algorithm...")
    start_time = time.time()
    track_to_cluster = r_matching_single_camera(
        corrected_tracklets, 
        frame_shape,
        distance_matrix,
        valid_track_ids=valid_track_ids,
        similarity_threshold=0.2,
        err_threshold=40
    )
    logging.info(f"R-matching completed in {time.time() - start_time:.2f} seconds")
    
    # Step 5: Save results
    logging.info("Saving results...")
    save_results(corrected_tracklets, track_to_cluster, final_output_file)
    saved_files = save_all_matching_data(
        output_dir, 
        corrected_tracklets, 
        track_to_cluster, 
        camera_id=None
    )
    
    # Step 6: Create visualizations
    distance_matrix_path = os.path.join(output_dir, "distance_matrix.png")
    visualize_distance_matrix(corrected_tracklets, distance_matrix_path)
    
    logging.info(f"Processing completed for session {session_id}")
    logging.info(f"Results saved to: {final_output_file}")
    
    return final_output_file

def main():
    """
    Command line interface for offline processing
    """
    parser = argparse.ArgumentParser(description="Offline R-matching for multi-camera people tracking")
    parser.add_argument("--session_dir", type=str, required=True, 
                       help="Path to session directory containing tracking_results.txt and features/")
    parser.add_argument("--video", type=str, 
                       help="Path to video file (optional, for frame dimensions)")
    parser.add_argument("--visualize", action="store_true", 
                       help="Create visualization video of results")
    parser.add_argument("--log_file", type=str, default="/home/aidev/workspace/reid/Thesis/reid-2024/trash/2025-05-22 20-42-37-2de4f2989e1f4e4298b81da615925c83/example.log",
                       help="Path to log file containing ID switch information")
    
    args = parser.parse_args()
    
    session_dir = args.session_dir.strip()
    
    if not os.path.exists(session_dir):
        logging.error(f"Session directory not found: {session_dir}")
        return
    
    # Process the session
    final_output_file = process_session(session_dir, args.video)
    
    # Create visualization if requested
    if args.visualize and final_output_file and os.path.exists(final_output_file):
        output_dir = os.path.dirname(final_output_file)
        output_video = os.path.join(output_dir, "visualized_results.avi")
        visualize_results(session_dir, final_output_file, output_video)

if __name__ == "__main__":
    main()