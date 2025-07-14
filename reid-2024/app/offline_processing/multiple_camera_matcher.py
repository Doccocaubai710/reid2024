import os
import logging
import numpy as np
import json
from scipy.spatial.distance import cosine
from scipy.optimize import linear_sum_assignment
import pickle
import argparse
from collections import defaultdict
import time

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
    
    def assign_global_ids(self):
        """
        Assign global IDs to clusters across all cameras
        """
        # First, build all camera pairs
        camera_pairs = self.build_camera_pairs()
        
        # Dictionary to track which clusters have been matched
        matched_clusters = set()
        
        # First pass: process all camera pairs
        for camera_a, camera_b in camera_pairs:
            matches = self.match_cameras(camera_a, camera_b)
            
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
    
    # Replace the save_results method in testv1.py with this simplified version
    # Replace these methods in testv1.py to skip JSON saving and focus on final results only

    def save_cross_camera_analysis(self, output_dir):
        """
        Compute cross-camera distance analysis and save only essential results
        """
        # Compute cross-camera distances (but don't save the JSON)
        cross_camera_distances = self.compute_and_store_cross_camera_distances()
        
        # Identify outliers
        outliers = self.identify_outlier_tracklets(cross_camera_distances)
        
        # Save only a simple text summary (no JSON)
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
        
        logging.info(f"Cross-camera analysis summary saved to {summary_file}")
        
        return {
            'summary_file': summary_file
        }

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
                        
                        # Write with global ID in MOT format
                        # Format: frame, global_id, x, y, w, h, conf, -1, -1, -1
                        conf = 1.0  # Default confidence
                        out_f.write(f"{frame},{global_id},{x},{y},{w},{h},{conf},-1,-1,-1\n")
                        
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
        
        # # Save cross-camera analysis (text summary only)
        # logging.info("Performing cross-camera distance analysis...")
        # try:
        #     analysis_files = self.save_cross_camera_analysis(output_dir)
        #     logging.info("Cross-camera analysis completed")
        # except Exception as e:
        #     logging.warning(f"Cross-camera analysis failed: {e}. Continuing with matching...")
        #     analysis_files = {}
        
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





def extract_cluster_features(base_dir, camera_dirs=None):
    """
    Extract representative features for each cluster in each camera
    
    Args:
        base_dir: Base directory containing camera result folders
        camera_dirs: List of camera directory names (if None, auto-detect)
    """
    # This would require access to the tracklet features, which might not be available
    # For a complete implementation, you'd need to:
    # 1. Load tracklet features for each camera
    # 2. Use cluster assignments to group features
    # 3. Compute a representative feature for each cluster (e.g., average)
    # 4. Save cluster features to a pickle file
    
    logging.warning("Feature extraction not implemented - requires access to tracklet features")
    logging.warning("Implement this function if features are available but not pre-computed per cluster")


def main():
    parser = argparse.ArgumentParser(description="Multi-camera tracklet matching")
    parser.add_argument("--base_dir", type=str, required=True, help="Base directory containing camera result folders")
    parser.add_argument("--cameras", type=str, nargs='+', help="List of camera directory names (optional)")
    parser.add_argument("--output_dir", type=str, default="multi_camera_results", help="Directory to save results")
    parser.add_argument("--similarity_threshold", type=float, default=0.3, help="Threshold for considering a match valid")
    parser.add_argument("--temporal_threshold", type=int, default=5000, help="Temporal constraint for matching (in frame units)")
    parser.add_argument("--extract_features", action="store_true", help="Extract cluster features before matching")
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