import os
import logging
import numpy as np
import json
from scipy.spatial.distance import cosine
from scipy.optimize import linear_sum_assignment
from scipy.cluster.hierarchy import fcluster, linkage
import pickle
import argparse
from collections import defaultdict
import time
from concurrent.futures import ProcessPoolExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("multi_camera_matching.log"),
        logging.StreamHandler()
    ]
)

class MultiCameraMatching:
    def __init__(self, base_dir, camera_dirs=None, similarity_threshold=0.3, temporal_threshold=5000):
        """
        Initialize multi-camera matching module
        
        Args:
            base_dir: Base directory containing camera result folders
            camera_dirs: List of camera directory names (if None, auto-detect)
            similarity_threshold: Threshold for considering a match valid
            temporal_threshold: Temporal constraint for matching (in frame units)
        """
        self.base_dir = base_dir
        self.similarity_threshold = similarity_threshold
        self.temporal_threshold = temporal_threshold
        
        # Auto-detect camera directories if not provided
        if camera_dirs is None:
            self.camera_dirs = self._detect_camera_dirs()
        else:
            self.camera_dirs = camera_dirs
            
        self.camera_clusters = {}  # Dict to store clusters per camera
        self.camera_features = {}  # Dict to store representative features per camera cluster
        self.camera_frames = {}    # Dict to store frame ranges per camera cluster
        self.global_ids = {}       # Dict to map camera_id:cluster_id to global_id
        self.next_global_id = 0    # Counter for assigning global IDs
        
        logging.info(f"Initialized MultiCameraMatching with {len(self.camera_dirs)} cameras")
        logging.info(f"Cameras: {self.camera_dirs}")

    def _detect_camera_dirs(self):
        """Auto-detect camera result directories based on folder structure"""
        camera_dirs = []
        for item in os.listdir(self.base_dir):
            if os.path.isdir(os.path.join(self.base_dir, item)):
                if os.path.exists(os.path.join(self.base_dir, item, "offline_results", "final_results.txt")):
                    camera_dirs.append(item)
        return camera_dirs
    def load_camera_data(self):
        """Load tracking and feature data from all cameras"""
        for camera in self.camera_dirs:
            camera_path = os.path.join(self.base_dir, camera)
            
            results_file = os.path.join(camera_path, "offline_results", "final_results.txt")
            clusters_file = os.path.join(camera_path, "offline_results","matching_data", "clusters.json")
            features_file = os.path.join(camera_path, "offline_results","matching_data", "cluster_features.pkl")
            
            # Check if files exist
            if not os.path.exists(results_file):
                logging.warning(f"Results file not found for camera {camera}: {results_file}")
                continue
                
            # Load cluster assignments and frame ranges
            try:
                self._load_clusters(camera, results_file)
                logging.info(f"Loaded tracking results for camera {camera} with {len(self.camera_clusters[camera])} clusters")
            except Exception as e:
                logging.error(f"Error loading tracking results for camera {camera}: {e}")
                continue
                
            # Load cluster features
            try:
                if os.path.exists(features_file):
                    self._load_features(camera, features_file)
                    logging.info(f"Loaded feature data from {features_file}")
                else:
                    # Attempt to extract features from clusters.json if available
                    if os.path.exists(clusters_file):
                        self._extract_features_from_clusters(camera, clusters_file)
                        logging.info(f"Extracted features from {clusters_file}")
                    else:
                        logging.warning(f"No feature data found for camera {camera}")
                        continue
            except Exception as e:
                logging.error(f"Error loading feature data for camera {camera}: {e}")
                continue
        
        logging.info(f"Loaded data for {len(self.camera_clusters)} cameras")
        
    def _load_clusters(self, camera, results_file):
        """
        Load cluster assignments and frame ranges from tracking results
        
        Args:
            camera: Camera ID
            results_file: Path to final_results.txt
        """
        self.camera_clusters[camera] = {}
        self.camera_frames[camera] = {}
        
        # Read tracking results and extract cluster info
        cluster_frames = defaultdict(list)
        
        with open(results_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 7:
                    continue
                    
                # Format: camera_id, cluster_id, frame, x, y, w, h, -1, -1
                # We're interested in cluster_id and frame
                cluster_id = int(parts[1])
                frame = int(parts[2])
                
                # Add to frame list for this cluster
                cluster_frames[cluster_id].append(frame)
        
        # Process frame ranges for each cluster
        for cluster_id, frames in cluster_frames.items():
            # Skip invalid cluster IDs (like -1)
            if cluster_id < 0:
                continue
                
            # Store cluster info
            self.camera_clusters[camera][cluster_id] = frames
            
            # Calculate frame range
            min_frame = min(frames)
            max_frame = max(frames)
            self.camera_frames[camera][cluster_id] = (min_frame, max_frame)
    
    def _load_features(self, camera, features_file):
        """
        Load precomputed cluster features from pickle file
        
        Args:
            camera: Camera ID
            features_file: Path to cluster features pickle file
        """
        with open(features_file, 'rb') as f:
            self.camera_features[camera] = pickle.load(f)
    
    def _extract_features_from_clusters(self, camera, clusters_file):
        """
        Extract feature representations from clusters JSON file
        
        Args:
            camera: Camera ID
            clusters_file: Path to clusters.json file
        """
        with open(clusters_file, 'r') as f:
            clusters = json.load(f)
        
        self.camera_features[camera] = {}
        
        # Process each cluster
        for cluster_id, tracklet_ids in clusters.items():
            # This would need access to actual feature data, which may not be in the JSON
            # For now, we'll just log a warning
            logging.warning(f"Feature extraction from clusters not implemented for camera {camera}")
            return
    def load_ensemble_features(self, camera):
        """
        Load and combine features from multiple models for improved cross-domain matching
        
        Args:
            camera: Camera ID
            
        Returns:
            Dictionary of combined features for each cluster
        """
        # Check if alternative feature files exist
        transreid_file = os.path.join(self.base_dir, camera, "offline_results", "matching_data", "transreid_features.pkl")
        hrnet_file = os.path.join(self.base_dir, camera, "offline_results", "matching_data", "hrnet_features.pkl")
        
        transreid_features = {}
        hrnet_features = {}
        
        # Load TransReID features if available
        if os.path.exists(transreid_file):
            try:
                with open(transreid_file, 'rb') as f:
                    transreid_features = pickle.load(f)
                logging.info(f"Loaded TransReID features for camera {camera}")
            except Exception as e:
                logging.warning(f"Error loading TransReID features for camera {camera}: {e}")
        
        # Load HRNet features if available
        if os.path.exists(hrnet_file):
            try:
                with open(hrnet_file, 'rb') as f:
                    hrnet_features = pickle.load(f)
                logging.info(f"Loaded HRNet features for camera {camera}")
            except Exception as e:
                logging.warning(f"Error loading HRNet features for camera {camera}: {e}")
        
        # If both feature types are available, create a weighted combination
        if transreid_features and hrnet_features:
            combined_features = {}
            
            # Determine if this is likely a real or synthetic camera based on camera ID
            # This is a simple heuristic and might need to be adjusted
            is_real = "real" in camera.lower() or "S001" in camera
            
            # Set weights based on domain
            transreid_weight = 0.7 if is_real else 0.5
            hrnet_weight = 0.3 if is_real else 0.5
            
            for cluster_id in self.camera_clusters[camera]:
                str_cluster_id = str(cluster_id)
                if str_cluster_id in transreid_features and str_cluster_id in hrnet_features:
                    combined_features[str_cluster_id] = (
                        transreid_weight * transreid_features[str_cluster_id] + 
                        hrnet_weight * hrnet_features[str_cluster_id]
                    )
            
            logging.info(f"Created ensemble features for camera {camera} with {len(combined_features)} clusters")
            return combined_features
        
        # If only one feature type is available, return it
        if transreid_features:
            return transreid_features
        if hrnet_features:
            return hrnet_features
        
        # If no alternative features are available, return None
        return None
    
    def build_camera_pairs(self):
        """Build all possible camera pairs for matching"""
        camera_pairs = []
        cameras = list(self.camera_clusters.keys())
        
        for i in range(len(cameras)):
            for j in range(i+1, len(cameras)):
                camera_pairs.append((cameras[i], cameras[j]))
                
        return camera_pairs
    
    def compute_distance_matrix(self, camera_a, camera_b):
        """
        Compute distance matrix between clusters from two cameras
        
        Args:
            camera_a: First camera ID
            camera_b: Second camera ID
            
        Returns:
            distance_matrix: 2D numpy array of distances
            valid_matches: 2D boolean array of valid matches (considering temporal constraints)
        """
        # Get clusters and features for both cameras
        clusters_a = self.camera_clusters[camera_a]
        clusters_b = self.camera_clusters[camera_b]
        
        features_a = self.camera_features[camera_a]
        features_b = self.camera_features[camera_b]
        
        frames_a = self.camera_frames[camera_a]
        frames_b = self.camera_frames[camera_b]
        
        # Create lists of cluster IDs
        cluster_ids_a = list(clusters_a.keys())
        cluster_ids_b = list(clusters_b.keys())
        
        # Initialize distance matrix and valid matches matrix
        n_clusters_a = len(cluster_ids_a)
        n_clusters_b = len(cluster_ids_b)
        
        distance_matrix = np.ones((n_clusters_a, n_clusters_b))
        valid_matches = np.ones((n_clusters_a, n_clusters_b), dtype=bool)
        
        # Compute distances between all cluster pairs
        for i, cluster_id_a in enumerate(cluster_ids_a):
            for j, cluster_id_b in enumerate(cluster_ids_b):
                # Check temporal constraint
                range_a = frames_a[cluster_id_a]
                range_b = frames_b[cluster_id_b]
                
                # If frame ranges overlap substantially, this is likely not the same person
                # (unless cameras have overlapping fields of view)
                overlap = min(range_a[1], range_b[1]) - max(range_a[0], range_b[0])
                
                if overlap > self.temporal_threshold:
                    valid_matches[i, j] = False
                    distance_matrix[i, j] = 2.0  # Set a high distance
                    continue
                
                # Compute feature distance
                if str(cluster_id_a) in features_a and str(cluster_id_b) in features_b:
                    feature_a = features_a[str(cluster_id_a)]
                    feature_b = features_b[str(cluster_id_b)]
                    
                    # Ensure features have the same shape
                    if isinstance(feature_a, np.ndarray) and isinstance(feature_b, np.ndarray):
                        if feature_a.shape == feature_b.shape:
                            # Compute cosine distance
                            distance_matrix[i, j] = cosine(feature_a, feature_b)
        
        return distance_matrix, valid_matches, cluster_ids_a, cluster_ids_b
    def determine_dynamic_threshold(self):
        """
        Determine dynamic threshold based on distance distributions
        
        Returns:
            Adjusted similarity threshold
        """
        intra_camera_distances = []
        inter_camera_distances = []
        
        # Collect some sample distances to analyze distributions
        sample_count = 0
        max_samples = 1000  # Limit samples to avoid excessive computation
        
        # Sample inter-camera distances
        for camera_a, camera_b in self.build_camera_pairs():
            if sample_count >= max_samples:
                break
                
            # Compute distance matrix
            try:
                distance_matrix, valid_matches, _, _ = self.compute_distance_matrix(camera_a, camera_b)
                
                # Extract valid distances
                valid_distances = distance_matrix[valid_matches]
                inter_camera_distances.extend(valid_distances.flatten().tolist())
                
                sample_count += len(valid_distances)
            except Exception as e:
                logging.warning(f"Error computing distances between {camera_a} and {camera_b}: {e}")
                continue
        
        # If we have enough samples, adjust the threshold
        if len(inter_camera_distances) > 10:
            inter_mean = np.mean(inter_camera_distances)
            inter_std = np.std(inter_camera_distances)
            
            # Heuristic: base threshold on distribution statistics
            # Lower mean distance suggests more similar appearances across cameras
            base_threshold = self.similarity_threshold
            
            # Adjust threshold based on mean and standard deviation
            if inter_mean < 0.3:  # Very similar appearances
                adjusted_threshold = base_threshold * 1.1  # More permissive
            elif inter_mean > 0.7:  # Very different appearances
                adjusted_threshold = base_threshold * 0.9  # More strict
            else:
                adjusted_threshold = base_threshold
            
            # Further adjust based on standard deviation
            # High std deviation suggests more uncertainty, be more conservative
            if inter_std > 0.2:
                adjusted_threshold *= 0.95
            
            # Ensure threshold stays within reasonable bounds
            adjusted_threshold = max(0.2, min(0.5, adjusted_threshold))
            
            logging.info(f"Dynamic threshold adjustment: {base_threshold:.3f} -> {adjusted_threshold:.3f} " +
                        f"(mean: {inter_mean:.3f}, std: {inter_std:.3f})")
            
            return adjusted_threshold
        
        # If not enough samples, return original threshold
        return self.similarity_threshold
    
    def match_cameras(self, camera_a, camera_b):
        """
        Match clusters between two cameras
        
        Args:
            camera_a: First camera ID
            camera_b: Second camera ID
            
        Returns:
            matches: List of (cluster_id_a, cluster_id_b) matches
        """
        logging.info(f"Matching cameras {camera_a} and {camera_b}")
        
        # Compute distance matrix
        distance_matrix, valid_matches, cluster_ids_a, cluster_ids_b = self.compute_distance_matrix(camera_a, camera_b)
        
        # Apply threshold to distance matrix
        thresholded_matrix = np.copy(distance_matrix)
        thresholded_matrix[~valid_matches] = 2.0  # Invalid matches get high distance
        thresholded_matrix[thresholded_matrix > self.similarity_threshold] = 2.0  # Apply similarity threshold
        
        # Use Hungarian algorithm for optimal assignment
        row_indices, col_indices = linear_sum_assignment(thresholded_matrix)
        
        # Extract valid matches
        matches = []
        for i, j in zip(row_indices, col_indices):
            if thresholded_matrix[i, j] < self.similarity_threshold:
                cluster_id_a = cluster_ids_a[i]
                cluster_id_b = cluster_ids_b[j]
                matches.append((cluster_id_a, cluster_id_b, distance_matrix[i, j]))
                logging.info(f"Match: Camera {camera_a} Cluster {cluster_id_a} <-> Camera {camera_b} Cluster {cluster_id_b} (distance: {distance_matrix[i, j]:.4f})")
        
        return matches
    
    def match_all_cameras_parallel(self):
        """
        Match all camera pairs in parallel
        
        Returns:
            Dictionary of matches per camera pair
        """
        camera_pairs = self.build_camera_pairs()
        all_matches = {}
        
        # Check if we have enough camera pairs to benefit from parallelization
        if len(camera_pairs) > 3:
            with ProcessPoolExecutor() as executor:
                futures = {executor.submit(self.match_cameras, camera_a, camera_b): (camera_a, camera_b) 
                          for camera_a, camera_b in camera_pairs}
                
                for future in futures:
                    try:
                        camera_a, camera_b = futures[future]
                        matches = future.result()
                        all_matches[(camera_a, camera_b)] = matches
                    except Exception as e:
                        logging.error(f"Error matching cameras: {e}")
        else:
            # For few cameras, sequential is simpler
            for camera_a, camera_b in camera_pairs:
                try:
                    matches = self.match_cameras(camera_a, camera_b)
                    all_matches[(camera_a, camera_b)] = matches
                except Exception as e:
                    logging.error(f"Error matching cameras {camera_a} and {camera_b}: {e}")
        
        return all_matches
    def assign_global_ids_with_clustering(self):
        """
        Assign global IDs using hierarchical clustering for improved global consistency
        """
        # Build list of all clusters and their features
        all_clusters = []
        cluster_features = []
        cluster_to_camera = {}
        cluster_to_idx = {}
        
        for camera in self.camera_clusters:
            for cluster_id in self.camera_clusters[camera]:
                key = (camera, cluster_id)
                all_clusters.append(key)
                
                # Get feature for this cluster
                str_cluster_id = str(cluster_id)
                if str_cluster_id in self.camera_features[camera]:
                    feature = self.camera_features[camera][str_cluster_id]
                    if isinstance(feature, np.ndarray):
                        cluster_features.append(feature)
                        cluster_to_camera[key] = camera
                        cluster_to_idx[key] = len(all_clusters) - 1
        
        n_clusters = len(all_clusters)
        if n_clusters == 0:
            logging.warning("No valid clusters found for hierarchical clustering")
            return
            
        logging.info(f"Building distance matrix for {n_clusters} clusters across all cameras")
        
        # Initialize distance matrix
        distance_matrix = np.ones((n_clusters, n_clusters))
        
        # Fill distance matrix - this can be computationally expensive for large datasets
        for i in range(n_clusters):
            camera_i, cluster_id_i = all_clusters[i]
            
            for j in range(i+1, n_clusters):  # Only compute upper triangle
                camera_j, cluster_id_j = all_clusters[j]
                
                # Same camera clusters can't match
                if camera_i == camera_j:
                    distance_matrix[i, j] = 2.0
                    distance_matrix[j, i] = 2.0
                    continue
                
                # Check temporal constraint
                range_i = self.camera_frames[camera_i][cluster_id_i]
                range_j = self.camera_frames[camera_j][cluster_id_j]
                
                overlap = min(range_i[1], range_j[1]) - max(range_i[0], range_j[0])
                
                if overlap > self.temporal_threshold:
                    distance_matrix[i, j] = 2.0
                    distance_matrix[j, i] = 2.0
                    continue
                
                # Compute feature distance
                feature_i = cluster_features[i]
                feature_j = cluster_features[j]
                
                if feature_i.shape == feature_j.shape:
                    dist = cosine(feature_i, feature_j)
                    distance_matrix[i, j] = dist
                    distance_matrix[j, i] = dist
        
        # Convert to condensed distance matrix (upper triangle only, required for linkage)
        condensed_dist = []
        for i in range(n_clusters):
            for j in range(i+1, n_clusters):
                condensed_dist.append(distance_matrix[i, j])
        
        # Apply hierarchical clustering
        try:
            Z = linkage(condensed_dist, method='complete')
            cluster_labels = fcluster(Z, self.similarity_threshold, criterion='distance')
            
            # Assign global IDs based on cluster labels
            for i, (camera, cluster_id) in enumerate(all_clusters):
                key = f"{camera}:{cluster_id}"
                self.global_ids[key] = int(cluster_labels[i])
                
            self.next_global_id = max(cluster_labels) + 1
            logging.info(f"Assigned {self.next_global_id - 1} global IDs using hierarchical clustering")
            
        except Exception as e:
            logging.error(f"Error in hierarchical clustering: {e}")
            logging.info("Falling back to pairwise matching approach")
            self.assign_global_ids()
    
    def assign_global_ids(self):
        """
        Assign global IDs to clusters across all cameras using pairwise matching
        """
        # First, build all camera pairs
        camera_pairs = self.build_camera_pairs()
        
        # Dictionary to track which clusters have been matched
        matched_clusters = set()
        
        # Matches from all camera pairs
        all_matches = self.match_all_cameras_parallel()
        
        # First pass: process all camera pairs
        for (camera_a, camera_b), matches in all_matches.items():
            for cluster_id_a, cluster_id_b, distance in matches:
                key_a = f"{camera_a}:{cluster_id_a}"
                key_b = f"{camera_b}:{cluster_id_b}"
                
                # Check if either cluster has been assigned a global ID
                if key_a in self.global_ids:
                    # Assign same global ID to cluster_b
                    self.global_ids[key_b] = self.global_ids[key_a]
                elif key_b in self.global_ids:
                    # Assign same global ID to cluster_a
                    self.global_ids[key_a] = self.global_ids[key_b]
                else:
                    # Both clusters are new, assign a new global ID
                    self.global_ids[key_a] = self.next_global_id
                    self.global_ids[key_b] = self.next_global_id
                    self.next_global_id += 1
                
                # Mark clusters as matched
                matched_clusters.add(key_a)
                matched_clusters.add(key_b)
        
        # Second pass: assign global IDs to unmatched clusters
        for camera in self.camera_clusters:
            for cluster_id in self.camera_clusters[camera]:
                key = f"{camera}:{cluster_id}"
                
                if key not in self.global_ids:
                    self.global_ids[key] = self.next_global_id
                    self.next_global_id += 1
        
        logging.info(f"Assigned {self.next_global_id} global IDs across {len(self.camera_clusters)} cameras")
    def refine_global_ids(self):
        """
        Refine global ID assignments based on confidence and consistency
        """
        # Calculate average distance for each global ID cluster
        global_id_distances = defaultdict(list)
        global_id_members = defaultdict(list)
        
        # Collect data about global ID clusters
        for camera_a, camera_b in self.build_camera_pairs():
            # Compute distance matrix
            try:
                distance_matrix, valid_matches, cluster_ids_a, cluster_ids_b = self.compute_distance_matrix(camera_a, camera_b)
                
                # For each pair of clusters
                for i, cluster_id_a in enumerate(cluster_ids_a):
                    for j, cluster_id_b in enumerate(cluster_ids_b):
                        key_a = f"{camera_a}:{cluster_id_a}"
                        key_b = f"{camera_b}:{cluster_id_b}"
                        
                        # If they have the same global ID
                        if key_a in self.global_ids and key_b in self.global_ids and self.global_ids[key_a] == self.global_ids[key_b]:
                            global_id = self.global_ids[key_a]
                            
                            # Record the distance
                            if valid_matches[i, j]:
                                global_id_distances[global_id].append(distance_matrix[i, j])
                                global_id_members[global_id].append((key_a, key_b))
            
            except Exception as e:
                logging.warning(f"Error computing distances for refinement between {camera_a} and {camera_b}: {e}")
                continue
        
        # Calculate average distance for each global ID
        avg_distances = {}
        for global_id, distances in global_id_distances.items():
            if distances:
                avg_distances[global_id] = np.mean(distances)
        
        # Identify and fix outliers
        refined_count = 0
        for global_id, members in global_id_members.items():
            if global_id not in avg_distances:
                continue
                
            avg_dist = avg_distances[global_id]
            threshold = 1.5 * avg_dist  # Outlier threshold
            
            # Check each pair
            for key_a, key_b in members:
                camera_a, cluster_id_a = key_a.split(':')
                camera_b, cluster_id_b = key_b.split(':')
                cluster_id_a = int(cluster_id_a)
                cluster_id_b = int(cluster_id_b)
                
                # Get indices in distance matrix
                try:
                    _, _, cluster_ids_a, cluster_ids_b = self.compute_distance_matrix(camera_a, camera_b)
                    i = cluster_ids_a.index(cluster_id_a)
                    j = cluster_ids_b.index(cluster_id_b)
                    
                    distance_matrix, valid_matches, _, _ = self.compute_distance_matrix(camera_a, camera_b)
                    
                    if valid_matches[i, j] and distance_matrix[i, j] > threshold:
                        # This is an outlier, assign a new global ID
                        logging.info(f"Refinement: Outlier detected for global ID {global_id}, " +
                                   f"distance {distance_matrix[i, j]:.4f} > threshold {threshold:.4f}")
                        
                        # Determine which one to reassign (prefer the one with worse avg distance to others)
                        # This is a simplified heuristic - a more sophisticated approach would consider all distances
                        self.global_ids[key_b] = self.next_global_id
                        self.next_global_id += 1
                        refined_count += 1
                
                except (ValueError, IndexError, Exception) as e:
                    logging.warning(f"Error during refinement: {e}")
                    continue
        
        logging.info(f"Refined {refined_count} global ID assignments")
    
    

    def compute_and_store_cross_camera_distances(self):
        """
        Compute pairwise distances between tracklets from different cameras
        and store for outlier analysis
        
        Returns:
            Dictionary containing distance statistics and raw distances
        """
        logging.info("Computing cross-camera tracklet distances for outlier analysis")
        
        cross_camera_distances = {
            'pairwise_distances': {},  # Raw distances between camera pairs
            'statistics': {},          # Statistics per camera pair
            'tracklet_distances': {},  # All distances per tracklet
            'detailed_matches': {},    # Detailed matching information
            'metadata': {
                'cameras': list(self.camera_clusters.keys()),
                'similarity_threshold': self.similarity_threshold,
                'temporal_threshold': self.temporal_threshold
            }
        }
        
        # Build camera pairs
        camera_pairs = self.build_camera_pairs()
        
        for camera_a, camera_b in camera_pairs:
            logging.info(f"Computing distances between {camera_a} and {camera_b}")
            
            try:
                # Compute distance matrix
                distance_matrix, valid_matches, cluster_ids_a, cluster_ids_b = self.compute_distance_matrix(camera_a, camera_b)
                
                # Store raw pairwise distances with proper JSON serialization
                pair_key = f"{camera_a}_{camera_b}"
                cross_camera_distances['pairwise_distances'][pair_key] = {
                    'camera_a': camera_a,
                    'camera_b': camera_b,
                    'cluster_ids_a': [int(x) for x in cluster_ids_a],  # Convert to int
                    'cluster_ids_b': [int(x) for x in cluster_ids_b],  # Convert to int
                    'distance_matrix': distance_matrix.tolist(),
                    'valid_matches': valid_matches.astype(bool).tolist()  # Convert numpy bool to Python bool
                }
                
                # Extract valid distances for statistics
                valid_distances = distance_matrix[valid_matches]
                
                # Create detailed matching information
                detailed_matches = []
                no_match_info = []
                
                if len(valid_distances) > 0:
                    # Compute statistics
                    stats = {
                        'count': int(len(valid_distances)),  # Convert to int
                        'mean': float(np.mean(valid_distances)),
                        'std': float(np.std(valid_distances)),
                        'min': float(np.min(valid_distances)),
                        'max': float(np.max(valid_distances)),
                        'median': float(np.median(valid_distances)),
                        'percentile_25': float(np.percentile(valid_distances, 25)),
                        'percentile_75': float(np.percentile(valid_distances, 75)),
                        'below_threshold': int(np.sum(valid_distances < self.similarity_threshold)),
                        'threshold_ratio': float(np.sum(valid_distances < self.similarity_threshold) / len(valid_distances))
                    }
                    
                    cross_camera_distances['statistics'][pair_key] = stats
                    
                    # Store per-tracklet distances for outlier detection and create detailed logs
                    for i, cluster_id_a in enumerate(cluster_ids_a):
                        tracklet_key_a = f"{camera_a}:{cluster_id_a}"
                        if tracklet_key_a not in cross_camera_distances['tracklet_distances']:
                            cross_camera_distances['tracklet_distances'][tracklet_key_a] = {}
                        
                        # Store distances to all tracklets in camera_b
                        camera_b_distances = []
                        min_distance = float('inf')
                        best_match = None
                        
                        for j, cluster_id_b in enumerate(cluster_ids_b):
                            tracklet_key_b = f"{camera_b}:{cluster_id_b}"
                            distance = distance_matrix[i, j]
                            is_valid = bool(valid_matches[i, j])  # Convert to Python bool
                            is_match = distance < self.similarity_threshold
                            
                            if is_valid:
                                camera_b_distances.append({
                                    'target_tracklet': tracklet_key_b,
                                    'distance': float(distance),
                                    'is_match': is_match,
                                    'is_valid': is_valid
                                })
                                
                                # Track best match for detailed logging
                                if distance < min_distance:
                                    min_distance = distance
                                    best_match = {
                                        'target_tracklet': tracklet_key_b,
                                        'distance': float(distance),
                                        'is_match': is_match
                                    }
                        
                        cross_camera_distances['tracklet_distances'][tracklet_key_a][camera_b] = camera_b_distances
                        
                        # Log detailed matching information
                        if best_match:
                            if best_match['is_match']:
                                logging.info(f"  MATCH: {tracklet_key_a} -> {best_match['target_tracklet']} "
                                        f"(distance: {best_match['distance']:.3f})")
                                detailed_matches.append({
                                    'source': tracklet_key_a,
                                    'target': best_match['target_tracklet'],
                                    'distance': best_match['distance'],
                                    'status': 'matched'
                                })
                            else:
                                logging.info(f"  NO MATCH: {tracklet_key_a} -> best: {best_match['target_tracklet']} "
                                        f"(distance: {best_match['distance']:.3f} > threshold: {self.similarity_threshold:.3f})")
                                no_match_info.append({
                                    'source': tracklet_key_a,
                                    'best_candidate': best_match['target_tracklet'],
                                    'distance': best_match['distance'],
                                    'threshold': self.similarity_threshold,
                                    'status': 'no_match'
                                })
                        else:
                            logging.info(f"  NO VALID CANDIDATES: {tracklet_key_a} (temporal constraints)")
                            no_match_info.append({
                                'source': tracklet_key_a,
                                'best_candidate': None,
                                'distance': float('inf'),
                                'threshold': self.similarity_threshold,
                                'status': 'no_valid_candidates'
                            })
                    
                    # Store detailed matching information
                    cross_camera_distances['detailed_matches'][pair_key] = {
                        'matches': detailed_matches,
                        'no_matches': no_match_info,
                        'summary': {
                            'total_tracklets_a': len(cluster_ids_a),
                            'total_tracklets_b': len(cluster_ids_b),
                            'matches_found': len(detailed_matches),
                            'no_matches': len(no_match_info),
                            'match_rate': len(detailed_matches) / len(cluster_ids_a) if cluster_ids_a else 0.0
                        }
                    }
                    
                    logging.info(f"  {pair_key}: {stats['count']} valid pairs, "
                            f"mean distance: {stats['mean']:.3f}, "
                            f"matches below threshold: {stats['below_threshold']}, "
                            f"match rate: {cross_camera_distances['detailed_matches'][pair_key]['summary']['match_rate']:.2%}")
            
            except Exception as e:
                logging.error(f"Error computing distances for {camera_a}-{camera_b}: {e}")
                continue
        
        return cross_camera_distances

    def identify_outlier_tracklets(self, cross_camera_distances, outlier_threshold_multiplier=2.0):
        """
        Identify tracklets that are outliers based on their distance patterns
        
        Args:
            cross_camera_distances: Output from compute_and_store_cross_camera_distances
            outlier_threshold_multiplier: How many standard deviations above mean to consider outlier
            
        Returns:
            Dictionary of outlier tracklets with analysis
        """
        outliers = {
            'tracklets': {},
            'summary': {},
            'threshold_multiplier': outlier_threshold_multiplier
        }
        
        # For each camera pair, identify outliers
        for pair_key, stats in cross_camera_distances['statistics'].items():
            camera_a, camera_b = pair_key.split('_', 1)  # Handle camera names with underscores
            
            # Calculate outlier threshold
            outlier_threshold = stats['mean'] + outlier_threshold_multiplier * stats['std']
            
            # Find tracklets with distances above threshold
            pair_outliers = []
            
            for tracklet_key, camera_distances in cross_camera_distances['tracklet_distances'].items():
                if camera_distances.get(camera_b):
                    tracklet_camera = tracklet_key.split(':')[0]
                    if tracklet_camera == camera_a:
                        # Check if this tracklet has unusually high distances to camera_b
                        distances_to_b = [d['distance'] for d in camera_distances[camera_b]]
                        
                        if distances_to_b:
                            min_distance = min(distances_to_b)
                            avg_distance = np.mean(distances_to_b)
                            
                            # Consider outlier if minimum distance is still high
                            if min_distance > outlier_threshold:
                                pair_outliers.append({
                                    'tracklet': tracklet_key,
                                    'min_distance': float(min_distance),  # Ensure JSON serializable
                                    'avg_distance': float(avg_distance),
                                    'distances_count': len(distances_to_b),
                                    'outlier_threshold': float(outlier_threshold)
                                })
                                
                                logging.info(f"  OUTLIER DETECTED: {tracklet_key} - "
                                        f"min_distance: {min_distance:.3f} > threshold: {outlier_threshold:.3f}")
            
            if pair_outliers:
                outliers['tracklets'][pair_key] = pair_outliers
                outliers['summary'][pair_key] = {
                    'count': len(pair_outliers),
                    'threshold': float(outlier_threshold),
                    'mean_distance': float(stats['mean']),
                    'std_distance': float(stats['std'])
                }
                
                logging.info(f"  Found {len(pair_outliers)} outlier tracklets for {pair_key}")
        
        return outliers

    def save_cross_camera_analysis(self, output_dir):
        """
        Compute and save cross-camera distance analysis
        """
        # Compute cross-camera distances
        cross_camera_distances = self.compute_and_store_cross_camera_distances()
        
        # Identify outliers
        outliers = self.identify_outlier_tracklets(cross_camera_distances)
        
        # Save results
        distances_file = os.path.join(output_dir, "cross_camera_distances.json")
        with open(distances_file, 'w') as f:
            json.dump(cross_camera_distances, f, indent=2)
        
        outliers_file = os.path.join(output_dir, "outlier_tracklets.json")
        with open(outliers_file, 'w') as f:
            json.dump(outliers, f, indent=2)
        
        # Save a detailed summary report
        summary_file = os.path.join(output_dir, "cross_camera_analysis_summary.txt")
        with open(summary_file, 'w') as f:
            f.write("CROSS-CAMERA DISTANCE ANALYSIS SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Camera Pair Statistics:\n")
            for pair_key, stats in cross_camera_distances['statistics'].items():
                f.write(f"\n{pair_key}:\n")
                f.write(f"  Valid pairs: {stats['count']}\n")
                f.write(f"  Mean distance: {stats['mean']:.3f}\n")
                f.write(f"  Std distance: {stats['std']:.3f}\n")
                f.write(f"  Matches below threshold ({self.similarity_threshold}): {stats['below_threshold']}\n")
                f.write(f"  Match ratio: {stats['threshold_ratio']:.3f}\n")
                
                # Add detailed matching information
                if pair_key in cross_camera_distances['detailed_matches']:
                    match_info = cross_camera_distances['detailed_matches'][pair_key]
                    f.write(f"  Match rate: {match_info['summary']['match_rate']:.2%}\n")
            
            f.write(f"\n\nDetailed Matching Results:\n")
            for pair_key, match_info in cross_camera_distances.get('detailed_matches', {}).items():
                f.write(f"\n{pair_key}:\n")
                
                f.write(f"  Successful Matches ({len(match_info['matches'])}):\n")
                for match in match_info['matches']:
                    f.write(f"    {match['source']} -> {match['target']} (dist: {match['distance']:.3f})\n")
                
                f.write(f"  Failed Matches ({len(match_info['no_matches'])}):\n")
                for no_match in match_info['no_matches']:
                    if no_match['best_candidate']:
                        f.write(f"    {no_match['source']} -> best: {no_match['best_candidate']} "
                            f"(dist: {no_match['distance']:.3f} > threshold: {no_match['threshold']:.3f})\n")
                    else:
                        f.write(f"    {no_match['source']} -> no valid candidates (temporal constraints)\n")
            
            f.write(f"\n\nOutlier Tracklets (threshold: mean + {outliers['threshold_multiplier']} * std):\n")
            total_outliers = 0
            for pair_key, pair_outliers in outliers['tracklets'].items():
                f.write(f"\n{pair_key}:\n")
                for outlier in pair_outliers:
                    f.write(f"  {outlier['tracklet']}: min_dist={outlier['min_distance']:.3f}, "
                        f"avg_dist={outlier['avg_distance']:.3f}, threshold={outlier['outlier_threshold']:.3f}\n")
                    total_outliers += 1
            
            f.write(f"\nTotal outlier tracklets: {total_outliers}\n")
        
        logging.info(f"Cross-camera analysis saved to {output_dir}")
        logging.info(f"  - Distances: {distances_file}")
        logging.info(f"  - Outliers: {outliers_file}")
        logging.info(f"  - Summary: {summary_file}")
        
        return {
            'distances_file': distances_file,
            'outliers_file': outliers_file,
            'summary_file': summary_file
        }
    # Modify the run method to include cross-camera analysis
    def save_results(self, output_dir):
        """
        Save final tracking results as separate files for each camera with global IDs
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Process each camera separately
        camera_stats = {}
        
        for camera in self.camera_dirs:
            camera_path = os.path.join(self.base_dir, camera)
            results_file = os.path.join(camera_path, "offline_results", "final_results.txt")
            
            if not os.path.exists(results_file):
                logging.warning(f"Results file not found for camera {camera}")
                continue
            
            # Create output file for this camera
            camera_name = os.path.basename(camera)
            output_file = os.path.join(output_dir, f"{camera_name}_final_tracking.txt")
            
            camera_global_ids = set()
            detection_count = 0
            
            with open(output_file, 'w') as out_f:
                with open(results_file, 'r') as in_f:
                    for line in in_f:
                        parts = line.strip().split(',')
                        if len(parts) < 7:
                            continue
                        
                        # Format: camera_id, cluster_id, frame, x, y, w, h, -1, -1
                        camera_id = parts[0]
                        cluster_id = int(parts[1])
                        frame = int(parts[2])
                        x, y, w, h = parts[3:7]
                        
                        # Skip invalid cluster IDs
                        if cluster_id < 0:
                            continue
                        
                        # Get global ID
                        key = f"{camera}:{cluster_id}"
                        global_id = self.global_ids.get(key, -1)
                        
                        if global_id == -1:
                            logging.warning(f"No global ID found for {key}")
                            continue
                        
                        # Write with original format: camera_id, global_id, frame, x, y, w, h, -1, -1
                        out_f.write(f"{camera_id},{global_id},{frame},{x},{y},{w},{h},-1,-1\n")
                        
                        camera_global_ids.add(global_id)
                        detection_count += 1
            
            camera_stats[camera_name] = {
                'unique_ids': len(camera_global_ids),
                'total_detections': detection_count,
                'output_file': output_file,
                'global_ids': list(camera_global_ids)
            }
            
            logging.info(f"Saved {camera_name}: {len(camera_global_ids)} unique IDs, {detection_count} detections -> {output_file}")
        
        # Create a simple summary
        summary_file = os.path.join(output_dir, "camera_summary.txt")
        with open(summary_file, 'w') as f:
            f.write("CAMERA-WISE TRACKING RESULTS SUMMARY\n")
            f.write("=" * 40 + "\n\n")
            
            # Per-camera statistics
            for camera_name, stats in camera_stats.items():
                f.write(f"{camera_name}:\n")
                f.write(f"  Output file: {os.path.basename(stats['output_file'])}\n")
                f.write(f"  Unique IDs: {stats['unique_ids']}\n")
                f.write(f"  Total detections: {stats['total_detections']}\n")
                f.write(f"  Global IDs: {sorted(stats['global_ids'])}\n\n")
            
            # Show cross-camera IDs
            all_global_ids = {}
            for camera_name, stats in camera_stats.items():
                for global_id in stats['global_ids']:
                    if global_id not in all_global_ids:
                        all_global_ids[global_id] = []
                    all_global_ids[global_id].append(camera_name)
            
            cross_camera_ids = {gid: cameras for gid, cameras in all_global_ids.items() if len(cameras) > 1}
            single_camera_ids = {gid: cameras for gid, cameras in all_global_ids.items() if len(cameras) == 1}
            
            f.write(f"GLOBAL ID DISTRIBUTION:\n")
            f.write(f"Total unique global IDs: {len(all_global_ids)}\n")
            f.write(f"Cross-camera IDs: {len(cross_camera_ids)}\n")
            f.write(f"Single-camera IDs: {len(single_camera_ids)}\n\n")
            
            if cross_camera_ids:
                f.write(f"Cross-camera ID details:\n")
                for global_id, cameras in sorted(cross_camera_ids.items()):
                    f.write(f"  Global ID {global_id}: {', '.join(cameras)}\n")
            
            f.write(f"\nSingle-camera IDs:\n")
            for global_id, cameras in sorted(single_camera_ids.items()):
                f.write(f"  Global ID {global_id}: {cameras[0]}\n")
        
        logging.info(f"Saved camera summary to {summary_file}")
        logging.info(f"Created individual tracking files for {len(camera_stats)} cameras")
        
        return camera_stats

    def run(self, output_dir):
        """
        Simplified run method - final results only
        """
        start_time = time.time()
        
        # Load camera data
        self.load_camera_data()
        
        # Check for ensemble features
        use_ensemble = False
        for camera in self.camera_dirs:
            ensemble_features = self.load_ensemble_features(camera)
            if ensemble_features:
                self.camera_features[camera] = ensemble_features
                use_ensemble = True
        
        if use_ensemble:
            logging.info("Using ensemble features for matching")
        
        # Determine dynamic threshold
        adjusted_threshold = self.determine_dynamic_threshold()
        if adjusted_threshold != self.similarity_threshold:
            self.similarity_threshold = adjusted_threshold
            logging.info(f"Using dynamically adjusted threshold: {self.similarity_threshold:.3f}")
        
        # Save cross-camera analysis (text summary only)
        logging.info("Performing cross-camera distance analysis...")
        try:
            analysis_files = self.save_cross_camera_analysis(output_dir)
            logging.info("Cross-camera analysis completed")
        except Exception as e:
            logging.warning(f"Cross-camera analysis failed: {e}. Continuing with matching...")
            analysis_files = {}
        
        # Assign global IDs
        logging.info("Assigning global IDs...")
        try:
            self.assign_global_ids_with_clustering()
        except Exception as e:
            logging.error(f"Error in hierarchical clustering: {e}")
            logging.info("Falling back to pairwise matching approach")
            self.assign_global_ids()
        
        # Refine global IDs
        logging.info("Refining global ID assignments...")
        self.refine_global_ids()
        
        # Save individual camera results
        logging.info("Saving individual camera tracking results...")
        camera_stats = self.save_results(output_dir)
        
        elapsed = time.time() - start_time
        logging.info(f"Multi-camera matching completed in {elapsed:.2f} seconds")
        
        # Print final summary
        total_global_ids = len(set(self.global_ids.values()))
        cross_camera_count = len([gid for gid in set(self.global_ids.values()) 
                                if len([k for k in self.global_ids.keys() 
                                        if self.global_ids[k] == gid]) > 1])
        
        logging.info(f"FINAL RESULTS:")
        logging.info(f"  Total global IDs: {total_global_ids}")
        logging.info(f"  Cross-camera IDs: {cross_camera_count}")
        logging.info(f"  Single-camera IDs: {total_global_ids - cross_camera_count}")
        logging.info(f"  Output directory: {output_dir}")
        
        return {
            'total_global_ids': total_global_ids,
            'cross_camera_ids': cross_camera_count,
            'output_dir': output_dir,
            'camera_stats': camera_stats
        }
    
        
def main():
    parser = argparse.ArgumentParser(description="Multi-camera tracklet matching")
    parser.add_argument("--base_dir", type=str, required=True, help="Base directory containing camera result folders")
    parser.add_argument("--cameras", type=str, nargs='+', help="List of camera directory names (optional)")
    parser.add_argument("--output_dir", type=str, default="multi_camera_results", help="Directory to save results")
    parser.add_argument("--similarity_threshold", type=float, default=0.3, help="Threshold for considering a match valid")
    parser.add_argument("--temporal_threshold", type=int, default=5000, help="Temporal constraint for matching (in frame units)")
    parser.add_argument("--extract_features", action="store_true", help="Extract cluster features before matching")
    parser.add_argument("--use_hierarchical", action="store_true", help="Use hierarchical clustering for global ID assignment")
    parser.add_argument("--dynamic_threshold", action="store_true", help="Dynamically adjust similarity threshold")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Extract cluster features if requested
    if args.extract_features:
        extract_cluster_features(args.base_dir, args.cameras)
    
    # Run multi-camera matching
    matcher = MultiCameraMatching(
        args.base_dir,
        args.cameras,
        args.similarity_threshold,
        args.temporal_threshold
    )
    
    matcher.run(args.output_dir)


if __name__ == "__main__":
    main()
                