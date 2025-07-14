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
from typing import List, Tuple, Dict
from shapely.geometry import Polygon
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("offline_processing.log"),
        logging.StreamHandler()
    ]
)

class Tracklet(object):
    """Tracklet class to store and process track information"""
    def __init__(self, track_id, camera_id):
        self.track_id = track_id
        self.camera_id = camera_id
        self.features = []  # list of features
        self.bboxes = []    # list of bboxes [x, y, w, h]
        self.scores = []    # list of confidence scores
        self.frames = []    # list of frame numbers
        self._mean_features = None
        self.backup_bboxes = None
        self.backup_features = None
        self.backup_scores = None
        self.backup_frames = None
        self.outPolygonList = None  # For storing uncertain zones
        self.is_reliable = True     # Flag to indicate if this is a reliable tracklet
        self.parent_id=None
        self.fullbody_embedding=None
        self.first_embedding=None
        self.last_embedding=None
        self.k_best=[]
        

    def add_detection(self, frame, bbox, feature, score):
        """Add a detection to this tracklet"""
        feature = feature / np.linalg.norm(feature)
        
        self.frames.append(frame)
        self.bboxes.append(bbox)
        self.features.append(feature)
        self.scores.append(score)
        self._mean_features = None  # Reset cached mean features

        if self.first_embedding is None and score > 0.7:
            self.first_embedding = feature

        # Update fullbody embeddings (all with score > 0.7)
        if score > 0.7:
            self.last_embedding = feature
            self.k_best.append((score, feature))

            if not hasattr(self, 'fullbody_pool'):
                self.fullbody_pool = []
            self.fullbody_pool.append(feature)

        # Maintain top-5 best features
        self.k_best.sort(reverse=True, key=lambda x: x[0])
        self.k_best = self.k_best[:5]

        # Update fullbody_embedding as mean of features with score > 0.7
        if hasattr(self, 'fullbody_pool') and self.fullbody_pool:
            self.fullbody_embedding = np.mean(self.fullbody_pool, axis=0)
            self.fullbody_embedding /= np.linalg.norm(self.fullbody_embedding)

        # Update k_best_embedding as mean of top-5 features
        if self.k_best:
            top_features = [f for _, f in self.k_best]
            self.k_best_embedding = np.mean(top_features, axis=0)
            self.k_best_embedding /= np.linalg.norm(self.k_best_embedding)

    
    def get_best_detection_indices(self, all_tracklets=None, enable=False, max_overlap_threshold=0.2, max_area_count=30, final_count=7, frame_shape=(1080,1920)):
        """
        Get indices of the best detections based on sequential filtering:
        1. Filter out detections in uncertain zones
        2. Filter to minimize overlap with other tracklets' boxes in the same frame
        3. Filter out detections with aspect ratios outside 1.7-3.5 range
        4. From remaining detections, select largest by area
        5. From those, select highest confidence
        
        Args:
            all_tracklets: List of all tracklet objects (including self)
            enable: Whether to enable overlap filtering with other tracklets
            max_overlap_threshold: Maximum allowed IoU between boxes from different tracklets
            max_area_count: Number of largest boxes to consider after filtering
            final_count: Final number of boxes to select
            frame_shape: (height, width) of the frame
            
        Returns:
            List of indices to the best detections
        """
        
        if not self.bboxes or not self.features or not self.scores:
            return []
        
        def is_in_uncertain_zone(bbox, frame_shape):
            """Check if bbox is in uncertain zone using simple coordinate checks"""
            height, width = frame_shape
            x1, y1, x2, y2 = bbox
            
            # Define margins (same as in define_reliable_uncertain_zones)
            margin_x = int(width * 0.1)     # 10% margin from sides
            margin_y_near = int(height * 0.1)   # 10% margin from bottom
            margin_y_far = int(height * 0.2)    # 20% margin from top

            if x1 < margin_x:
                return True
            
            # Right edge  
            if x2 > (width - margin_x):
                return True
            
            # Near edge (bottom)
            if y1 < margin_y_near:
                return True
            return False    
        
        # Helper function to calculate IoU
        def calculate_iou(box1, box2):
            # Convert to x1, y1, x2, y2 format
            box1_x1, box1_y1, box1_x2, box1_y2 = box1
            box2_x1, box2_y1, box2_x2, box2_y2 = box2
            
            # Calculate intersection area
            x_left = max(box1_x1, box2_x1)
            y_top = max(box1_y1, box2_y1)
            x_right = min(box1_x2, box2_x2)
            y_bottom = min(box1_y2, box2_y2)
            
            if x_right < x_left or y_bottom < y_top:
                return 0.0  # No intersection
            
            intersection_area = (x_right - x_left) * (y_bottom - y_top)
            
            # Calculate union area
            box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
            box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
            union_area = box1_area + box2_area - intersection_area
            
            if union_area == 0:
                return 0.0
                
            return intersection_area / union_area
        
        # Step 1: Filter out detections in uncertain zones
        zone_filtered_indices = []
        for i, bbox in enumerate(self.bboxes):
            if not is_in_uncertain_zone(bbox, frame_shape):
                zone_filtered_indices.append(i)
        
        # If no detections outside uncertain zones, use all as fallback
        if not zone_filtered_indices:
            zone_filtered_indices = list(range(len(self.bboxes)))
        
        # Step 2: Filter based on overlap with other tracklets
        low_overlap_indices = zone_filtered_indices
        if all_tracklets and enable:
            # Get the set of frames for this tracklet
            self_frames = set(self.frames)
            overlapping_tracklets = []
            for tracklet in all_tracklets:
                if tracklet.track_id == self.track_id:  # Skip self
                    continue
                    
                # Check if tracklet appears in any of the same frames
                other_frames = set(tracklet.frames)
                if self_frames.intersection(other_frames):
                    overlapping_tracklets.append(tracklet)
            
            # Create a mapping from frame to bounding boxes of other tracklets
            frame_to_other_bboxes = {}
            for tracklet in overlapping_tracklets:
                for i, frame in enumerate(tracklet.frames):
                    if frame in self_frames:  # Only consider frames that this tracklet appears in
                        if frame not in frame_to_other_bboxes:
                            frame_to_other_bboxes[frame] = []
                        frame_to_other_bboxes[frame].append(tracklet.bboxes[i])
            
            low_overlap_indices = []
            for i in zone_filtered_indices:
                frame = self.frames[i]
                bbox = self.bboxes[i]
                max_iou = 0.0
                
                # Check overlap with other tracklets' bboxes in the same frame
                if frame in frame_to_other_bboxes:
                    for other_bbox in frame_to_other_bboxes[frame]:
                        iou = calculate_iou(bbox, other_bbox)
                        max_iou = max(max_iou, iou)
                
                # Keep detection if overlap is below threshold
                if max_iou <= max_overlap_threshold:
                    low_overlap_indices.append(i)
        
        # Step 3: Filter by aspect ratio (1.7 to 3.5)
        aspect_ratio_indices = []
        for i in low_overlap_indices:
            bbox = self.bboxes[i]
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            
            if width <= 0:  # Skip invalid boxes
                continue
                
            # Calculate aspect ratio (height/width)
            aspect_ratio = height / width
            
            # Only keep detections with aspect ratio between 1.7 and 3.5
            if 1.7 <= aspect_ratio <= 3.5:
                aspect_ratio_indices.append(i)
        
        # If no detections with good aspect ratio, fall back to low_overlap_indices
        if not aspect_ratio_indices:
            aspect_ratio_indices = low_overlap_indices
        
        # Step 4: From remaining boxes, select largest by area
        area_with_index = [(i, (self.bboxes[i][2]-self.bboxes[i][0]) * (self.bboxes[i][3]-self.bboxes[i][1])) for i in aspect_ratio_indices]
        area_with_index.sort(key=lambda x: x[1], reverse=True)  # Sort by area, largest first
        
        # Take the largest boxes, up to max_area_count
        largest_indices = [x[0] for x in area_with_index[:min(max_area_count, len(area_with_index))]]
        
        # Step 5: From largest boxes, select highest confidence
        conf_with_index = [(i, self.scores[i]) for i in largest_indices]
        conf_with_index.sort(key=lambda x: x[1], reverse=True)  # Sort by confidence, highest first
        
        # Take the highest confidence boxes, up to final_count
        final_indices = [x[0] for x in conf_with_index[:min(final_count, len(conf_with_index))]]
        
        # Optional: Print aspect ratios of selected detections for debugging
        for i in final_indices:
            bbox = self.bboxes[i]
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            aspect_ratio = height / width if width > 0 else 0
            print(f"Selected detection at frame {self.frames[i]}: aspect ratio={aspect_ratio:.2f}, score={self.scores[i]:.2f}")
        
        return final_indices
    def mean_features(self, all_tracklets=None):
        """
        Calculate mean feature vector using features from the best detections,
        prioritizing full-body shots with appropriate aspect ratio.
        
        Args:
            all_tracklets: List of all tracklet objects (including self)
        """
        if self._mean_features is None and self.features:
            
            # Get indices of best detections with fullbody preference
            best_indices = self.get_best_detection_indices(
                all_tracklets=all_tracklets,
                enable=True,
                max_overlap_threshold=0,
                max_area_count=30,
                final_count=1  # Select more boxes for better averaging
            )
            
            selected_frames = []
            if best_indices:
                # Get the selected features
                selected_frames = [self.frames[i] for i in best_indices]
                selected_features = [self.features[i] for i in best_indices]
                
                # Log selected frames for debugging
                print(f"Track {self.track_id} selected frames: {selected_frames}")
                
                # For debugging, print aspect ratios of selected boxes
                for i in best_indices:
                    w, h = self.bboxes[i][2], self.bboxes[i][3]
                    aspect = h/w if w > 0 else 0
                    print(f"  Frame {self.frames[i]}: aspect ratio = {aspect:.2f}, score = {self.scores[i]:.2f}")
                
                features_array = np.array(selected_features)
                
                # Check if we need to reshape
                if len(features_array.shape) > 2:
                    original_shape = features_array.shape
                    features_array = features_array.reshape(original_shape[0], -1)
                    
                # Calculate mean of selected features
                self._mean_features = np.mean(features_array, axis=0)
            elif self.features:
                # Fallback if no best indices were found: use all features
                features_array = np.array(self.features)
                
                # Check if we need to reshape
                if len(features_array.shape) > 2:
                    original_shape = features_array.shape
                    features_array = features_array.reshape(original_shape[0], -1)
                
                # Calculate mean of all features
                self._mean_features = np.mean(features_array, axis=0)
            
            norm = np.linalg.norm(self._mean_features)
            self._mean_features = self._mean_features / (norm if norm > 0 else 1.0)
        
        return self._mean_features, selected_frames
    def refine_tracklets(self):
        """
        Remove bboxes that are in outzone/uncertain zone, remove its feature also
        """
        self.backup_bboxes = copy.deepcopy(self.bboxes)
        self.backup_features = copy.deepcopy(self.features)
        self.backup_scores = copy.deepcopy(self.scores)
        self.backup_frames = copy.deepcopy(self.frames)

        new_boxes, new_features, new_scores, new_frames = [], [], [], []
        for (i, bbox) in enumerate(self.bboxes):
            if self.is_outzone(bbox):
                continue
            new_boxes.append(bbox)
            new_features.append(self.features[i])
            new_scores.append(self.scores[i])
            new_frames.append(self.frames[i])
        self.bboxes = new_boxes
        self.features = new_features
        self.scores = new_scores
        self.frames = new_frames
        self._mean_features = None

    def is_outzone(self, bbox):
        """
        Check if bbox is in outzone/uncertain zone
        """
        if self.outPolygonList is None:
            return False
        for outPolygon in self.outPolygonList:
            if self.get_iou(bbox, outPolygon) > 0.5:
                return True
        return False


    def get_iou(self, bbox, pts):
        """Calculate IoU between bbox and polygon"""
        x1, y1, x2, y2 = bbox
        new_bbox = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        polygon1 = Polygon(new_bbox)
        polygon2 = Polygon(pts)
        intersect = polygon1.intersection(polygon2).area
        return intersect / polygon1.area if polygon1.area > 0 else 0
    def restore(self):
        """Restore original tracklet data"""
        if self.backup_bboxes is None:
            return
        self.bboxes = self.backup_bboxes
        self.features = self.backup_features
        self.scores = self.backup_scores
        self.frames = self.backup_frames
        self._mean_features = None
    def detect_exit_entrance_switch(self, frame_size=(1920, 1080), exit_threshold=0.15, similarity_threshold=0.25):
        """
        Detect when a tracklet exits the frame and another enters
        causing an ID switch at frame boundaries
        
        Args:
            frame_size: Tuple of (width, height) of the frame
            exit_threshold: Threshold distance from frame edge to consider as boundary
            similarity_threshold: Threshold for cosine distance to consider as ID switch
            
        Returns:
            Tuple of (has_switch, split_point)
        """
        width, height = frame_size
        boundary_width = width * exit_threshold
        boundary_height = height * exit_threshold
        
        # Not enough frames to analyze
        if len(self.frames) < 10:
            return False, -1
        
        # Extract centers and velocities for all frames
        centers = []
        for bbox in self.bboxes:
            x1, y1, x2, y2 = bbox
            centers.append(((x1+x2)/2, (y1 + y2)/2))
        
        # Calculate velocities (direction vectors)
        velocities = []
        for i in range(1, len(centers)):
            dx = centers[i][0] - centers[i-1][0]
            dy = centers[i][1] - centers[i-1][1]
            velocities.append((dx, dy))
        
        # Track points where object interacts with edge
        edge_interactions = []
        direction_changes = []
        
        # Step 1: Find edge interactions and direction changes
        for i in range(1, len(centers)-1):
            x, y = centers[i]
            
            # Check if near edge
            near_left = x < boundary_width
            near_right = x > width - boundary_width
            near_top = y < boundary_height
            near_bottom = y > height - boundary_height
            
            if near_left or near_right or near_top or near_bottom:
                edge_interactions.append(i)
            
            # Skip first frame for direction check
            if i < 2:
                continue
                
            # Check for significant direction change
            v_before = velocities[i-2]  # i-2 to i-1
            v_after = velocities[i-1]   # i-1 to i
            
            # Calculate dot product to find angle
            dot_product = v_before[0]*v_after[0] + v_before[1]*v_after[1]
            mag_before = np.sqrt(v_before[0]**2 + v_before[1]**2)
            mag_after = np.sqrt(v_after[0]**2 + v_after[1]**2)
            
            # Avoid division by zero
            if mag_before > 0.1 and mag_after > 0.1:
                cos_angle = dot_product / (mag_before * mag_after)
                angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
                angle_degrees = np.degrees(angle)
                
                # Check for direction reversal (angle > 90 degrees)
                if angle_degrees > 90:
                    direction_changes.append(i)
        
        # Step 2: Identify potential split points as frames where:
        # - Object is near edge AND has a direction change, OR
        # - Object is near edge and has a velocity magnitude change
        potential_splits = []
        
        for i in edge_interactions:
            # Check if this is also a direction change
            if i in direction_changes or i-1 in direction_changes or i+1 in direction_changes:
                potential_splits.append(i)
                continue
                
            # Check for velocity magnitude change if not a direction change
            if i > 1 and i < len(velocities):
                v_before = velocities[i-2]
                v_after = velocities[i-1]
                mag_before = np.sqrt(v_before[0]**2 + v_before[1]**2)
                mag_after = np.sqrt(v_after[0]**2 + v_after[1]**2)
                
                # Check for significant speed change (doubled or halved)
                if mag_after > 2*mag_before or mag_after < 0.5*mag_before:
                    potential_splits.append(i)
        
        # Step 3: For each potential split point, check feature similarity
        for split_point in potential_splits:
            # Need sufficient frames before and after to calculate similarity
            if split_point < 5 or split_point >= len(self.features) - 5:
                continue
            
            # Create "mini-tracklets" for before and after the split point
            features_before = self.features[:split_point]
            features_after = self.features[split_point:]
            
            # Convert to numpy arrays
            features_before_np = np.array(features_before)
            features_after_np = np.array(features_after)
            
            # Reshape if needed
            if len(features_before_np.shape) > 2:
                features_before_np = features_before_np.reshape(features_before_np.shape[0], -1)
            
            if len(features_after_np.shape) > 2:
                features_after_np = features_after_np.reshape(features_after_np.shape[0], -1)
            
            # Calculate mean features
            mean_before = np.mean(features_before_np, axis=0)
            mean_after = np.mean(features_after_np, axis=0)
            
            # Calculate cosine distance
            try:
                cos_distance = distance.cosine(mean_before, mean_after)
                
                # Log for debugging
                logging.info(f"Tracklet {self.track_id} - Split point {split_point} (frame {self.frames[split_point]}) - " 
                            f"Cosine distance: {cos_distance:.3f}")
                
                # If distance exceeds threshold, we have an ID switch
                if cos_distance > similarity_threshold:
                    # Determine which edge caused the exit/entrance
                    x, y = centers[split_point]
                    edge_type = "unknown"
                    
                    if x < boundary_width:
                        edge_type = "left"
                    elif x > width - boundary_width:
                        edge_type = "right"
                    elif y < boundary_height:
                        edge_type = "top"
                    elif y > height - boundary_height:
                        edge_type = "bottom"
                    
                    logging.info(f"Exit-entrance switch detected at {edge_type} edge, "
                                f"frame {self.frames[split_point]} with distance {cos_distance:.3f}")
                    return True, split_point
            
            except Exception as e:
                logging.error(f"Error calculating feature similarity: {e}")
        
        return False, -1

    def detect_id_switch(self, similarity_threshold=0.4):
        """Detect if this tracklet contains an ID switch using GMM"""
        if len(self.features) < 10:
            return False, []

        # Reshape features to 2D if needed
        features_array = np.array(self.features)
        original_shape = features_array.shape
        
        # Check if we need to reshape
        if len(original_shape) > 2:
            # Reshape to 2D: (n_samples, n_features)
            features_array = features_array.reshape(original_shape[0], -1)
            
        gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=0)
        gmm.fit(features_array)
        
        # Calculate distance between means
        mean1 = gmm.means_[0].reshape(-1)
        mean2 = gmm.means_[1].reshape(-1)
        cos_distance = distance.cosine(mean1, mean2)

        if cos_distance < similarity_threshold:
            return False, []

        cluster_labels = gmm.predict(features_array)
        return True, cluster_labels
    

    def find_split_point(self, labels):
        """Find the optimal point to split a tracklet with ID switch"""
        # Create arrays tracking consecutive runs of 0s and 1s
        runs_of_0 = []
        runs_of_1 = []
        current_run_0 = 0
        current_run_1 = 0
        
        for i, label in enumerate(labels):
            if label == 0:
                current_run_0 += 1
                current_run_1 = 0
            else:  # label == 1
                current_run_1 += 1
                current_run_0 = 0
                
            runs_of_0.append(current_run_0)
            runs_of_1.append(current_run_1)
        
        # Find index of longest run for each value
        max_run_0 = max(runs_of_0)
        max_run_1 = max(runs_of_1)
        
        # Get starting indices of longest runs
        if max_run_0 > 0:
            idx_0 = runs_of_0.index(max_run_0) - max_run_0 + 1
        else:
            idx_0 = 0
            
        if max_run_1 > 0:
            idx_1 = runs_of_1.index(max_run_1) - max_run_1 + 1
        else:
            idx_1 = 0
        
        # Return the maximum of the two starting indices
        return max(idx_0, idx_1)

    def split_track(self, labels):
        """Split tracklet into two based on GMM labels, ensuring each has enough features"""
        split_point = self.find_split_point(labels)
        
        # Ensure at least 5 frames in each split (minimum for meaningful representation)
        if split_point < 5 or len(self.features) - split_point < 5:
            return None, None

        track_a = Tracklet(self.track_id, self.camera_id)
        track_b = Tracklet(f"{self.track_id}_split", self.camera_id)

        # Get confidence-sorted indices for first part
        first_part_indices = list(range(split_point))
        first_part_indices.sort(key=lambda i: self.scores[i], reverse=True)
        # Keep top 10 or all if less than 10
        first_part_indices = first_part_indices[:min(10, len(first_part_indices))]
        
        # Get confidence-sorted indices for second part
        second_part_indices = list(range(split_point, len(self.features)))
        second_part_indices.sort(key=lambda i: self.scores[i], reverse=True)
        # Keep top 10 or all if less than 10
        second_part_indices = second_part_indices[:min(10, len(second_part_indices))]
        
        # Populate first tracklet with top confident features from first part
        for i in first_part_indices:
            track_a.add_detection(
                self.frames[i],
                self.bboxes[i],
                self.features[i],
                self.scores[i]
            )
        
        # Populate second tracklet with top confident features from second part
        for i in second_part_indices:
            track_b.add_detection(
                self.frames[i],
                self.bboxes[i],
                self.features[i],
                self.scores[i]
            )

        return track_a, track_b


def extract_id_switches_from_log(log_file="example.log"):
    """
    Extract ID switch information from the log file and print it to the console.
    
    Args:
        log_file: Path to the log file
        
    Returns:
        A list of dictionaries containing ID switch information
    """
    # Regular expression to match ID switch detection lines
    id_switch_pattern = re.compile(r'ID switch detected: Track (\d+) at frame (\d+) near (\w+) edge')
    
    # List to store extracted information
    switches = []
    
    try:
        with open(log_file, 'r') as f:
            for line in f:
                # Check if line contains ID switch information
                match = id_switch_pattern.search(line)
                if match:
                    # Extract timestamp from the beginning of the line if available
                    timestamp_match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})', line)
                    timestamp = timestamp_match.group(1) if timestamp_match else None
                    
                    # Extract track ID, frame number, and edge type
                    track_id = match.group(1)
                    frame = match.group(2)
                    edge = match.group(3)
                    
                    # Create a dictionary with the extracted information
                    switch_info = {
                        'timestamp': timestamp,
                        'track_id': track_id,
                        'frame': frame,
                        'edge': edge
                    }
                    
                    switches.append(switch_info)
                    
                    
                # Also look for GMM-based switch detections
                gmm_match = re.search(r'GMM detected ID switch in track (\d+)', line)
                if gmm_match:
                    track_id = gmm_match.group(1)
                    timestamp_match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})', line)
                    timestamp = timestamp_match.group(1) if timestamp_match else None
                    
                    switch_info = {
                        'timestamp': timestamp,
                        'track_id': track_id,
                        'method': 'GMM'
                    }
                    
                    switches.append(switch_info)
                    
                    if timestamp:
                        print(f"[{timestamp}] GMM detected ID switch in track {track_id}")
                    else:
                        print(f"GMM detected ID switch in track {track_id}")
                
                # Look for exit-entrance switches
                exit_match = re.search(r'Exit-entrance switch detected .* track (\d+) at frame (\d+)', line)
                if exit_match:
                    track_id = exit_match.group(1)
                    frame = exit_match.group(2)
                    timestamp_match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})', line)
                    timestamp = timestamp_match.group(1) if timestamp_match else None
                    
                    switch_info = {
                        'timestamp': timestamp,
                        'track_id': track_id,
                        'frame': frame,
                        'method': 'Exit-entrance'
                    }
                    
                    switches.append(switch_info)
                    
                    if timestamp:
                        print(f"[{timestamp}] Exit-entrance switch detected in track {track_id} at frame {frame}")
                    else:
                        print(f"Exit-entrance switch detected in track {track_id} at frame {frame}")
                        
        print(f"\nTotal ID switches found: {len(switches)}")
        return switches
        
    except FileNotFoundError:
        print(f"Error: Log file '{log_file}' not found")
        return []
    except Exception as e:
        print(f"Error processing log file: {e}")
        return []


def filter_nearby_switches(data: list[dict], frame_gap: int = 20) -> list[dict]:
    """
    Filters ID switches by keeping only the last one if multiple entries for the same track_id
    happen within a close frame range.
    
    Args:
        data (List[Dict]): List of ID switch records.
        frame_gap (int): Maximum frame difference to consider as "nearby".
    
    Returns:
        List[Dict]: Filtered list of ID switches.
    """
    from collections import defaultdict

    # Group by track_id
    grouped = defaultdict(list)
    for entry in data:
        grouped[entry['track_id']].append(entry)

    result = []

    for track_id, entries in grouped.items():
        # Sort entries by frame number
        sorted_entries = sorted(entries, key=lambda x: int(x['frame']))
        temp_group = []

        for entry in sorted_entries:
            if not temp_group:
                temp_group.append(entry)
            else:
                prev_frame = int(temp_group[-1]['frame'])
                curr_frame = int(entry['frame'])
                if curr_frame - prev_frame <= frame_gap:
                    temp_group.append(entry)
                else:
                    # Add the last one in the temp_group
                    result.append(temp_group[-1])
                    temp_group = [entry]
        # Add the last from the final group
        if temp_group:
            result.append(temp_group[-1])

    return sorted(result, key=lambda x: int(x['frame']))

def extract_tracklet_brightness_contrast_from_video(video_path: str, tracklets_dict: Dict[int, Tracklet]) -> Dict[int, Dict]:
    """
    Reprocess video to extract brightness and contrast for given tracklets
    
    Returns per-detection brightness/contrast data
    """
    # Initialize tracklet statistics storage
    tracklet_stats = {}
    
    # Create frame-to-tracklets mapping with detection indices
    frame_to_tracklets = defaultdict(list)
    for track_id, tracklet in tracklets_dict.items():
        for i, frame_num in enumerate(tracklet.frames):
            frame_to_tracklets[frame_num].append({
                'track_id': track_id,
                'bbox': tracklet.bboxes[i],
                'detection_index': i  # Track which detection this is
            })
    
    # Initialize per-tracklet storage
    for track_id in tracklets_dict:
        tracklet_stats[track_id] = {
            'brightnesses': [None] * len(tracklets_dict[track_id].frames),
            'contrasts': [None] * len(tracklets_dict[track_id].frames)
        }
    
    # Open video and process frames
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    frame_num = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logging.info(f"Processing {total_frames} frames from {video_path}")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process detections for this frame
        if frame_num in frame_to_tracklets:
            for detection in frame_to_tracklets[frame_num]:
                track_id = detection['track_id']
                bbox = detection['bbox']
                det_idx = detection['detection_index']
                
                # Extract person region and compute brightness/contrast
                x, y, w, h = [int(coord) for coord in bbox]
                
                # Ensure bbox is within frame boundaries
                frame_height, frame_width = frame.shape[:2]
                x = max(0, min(x, frame_width - 1))
                y = max(0, min(y, frame_height - 1))
                w = max(1, min(w, frame_width - x))
                h = max(1, min(h, frame_height - y))
                
                # Extract person region
                person_region = frame[y:y+h, x:x+w]
                
                if person_region.size > 0:
                    # Convert to grayscale
                    if len(person_region.shape) == 3:
                        gray_region = cv2.cvtColor(person_region, cv2.COLOR_BGR2GRAY)
                    else:
                        gray_region = person_region
                    
                    # Calculate brightness and contrast
                    brightness = np.mean(gray_region)
                    contrast = np.std(gray_region)
                    
                    # Store at the correct detection index
                    tracklet_stats[track_id]['brightnesses'][det_idx] = brightness
                    tracklet_stats[track_id]['contrasts'][det_idx] = contrast
        
        frame_num += 1
        
        # Progress indicator
        if frame_num % 1000 == 0:
            logging.info(f"Processed {frame_num}/{total_frames} frames ({frame_num/total_frames*100:.1f}%)")
    
    cap.release()
    
    # Clean up None values and compute summary statistics
    final_stats = {}
    for track_id, data in tracklet_stats.items():
        # Filter out None values
        brightnesses = [b for b in data['brightnesses'] if b is not None]
        contrasts = [c for c in data['contrasts'] if c is not None]
        
        final_stats[track_id] = {
            'brightnesses': brightnesses,
            'contrasts': contrasts,
            'average_brightness': float(np.mean(brightnesses)) if brightnesses else 0.0,
            'brightness_std': float(np.std(brightnesses)) if len(brightnesses) > 1 else 0.0,
            'average_contrast': float(np.mean(contrasts)) if contrasts else 0.0,
            'contrast_std': float(np.std(contrasts)) if len(contrasts) > 1 else 0.0,
            'frame_count': len(brightnesses)
        }
    
    logging.info(f"Extracted brightness/contrast for {len(final_stats)} tracklets")
    return final_stats

def load_tracking_results_to_tracklets(tracking_file: str) -> Dict[int, Tracklet]:
    """
    Load tracking results from file and create Tracklet objects
    
    Args:
        tracking_file: Path to tracking results file (format: frame,track_id,x,y,w,h,score)
        
    Returns:
        Dictionary mapping track_id to Tracklet objects
    """
    tracklets = {}
    
    with open(tracking_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 7:
                continue
                
            frame, track_id, x1, y1, x2, y2, score = parts[:7]
            frame = int(frame)
            track_id = int(track_id)
            bbox = [float(x1), float(y1), float(x2), float(y2)]
            score = float(score)
            
            # Create tracklet if it doesn't exist
            if track_id not in tracklets:
                tracklets[track_id] = Tracklet(track_id, "camera_1")  # Use a default camera ID
            
            # Add detection (note: we don't have features here, so pass None)
            tracklets[track_id].add_detection(frame, bbox, None, score)
    
    return tracklets

def process_session_brightness_contrast(session_dir: str, tracking_filename: str = "tracking_results.txt"):
    """
    Process a session directory to extract brightness/contrast for all tracklets
    
    Args:
        session_dir: Path to session directory
        tracking_filename: Name of tracking results file
        
    Returns:
        Dictionary with brightness/contrast statistics per tracklet
    """
    video_path = "/home/aidev/workspace/reid/Thesis/reid-2024/app/assets/videos/cam2.MOV"
    tracking_file = "/home/aidev/workspace/reid/Thesis/reid-2024/trash/2025-05-14 16-36-09-96e885062e534a7e8433bca9b49d0f08/tracking_results.txt"
    
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    if not os.path.exists(tracking_file):
        raise FileNotFoundError(f"Tracking file not found: {tracking_file}")
    
    # Load tracking results and convert to Tracklet objects
    tracklets_dict = load_tracking_results_to_tracklets(tracking_file)
    
    # Extract brightness/contrast
    brightness_data = extract_tracklet_brightness_contrast_from_video(video_path, tracklets_dict)
    
    # Compare best vs all detections for each tracklet
    for track_id, tracklet in tracklets_dict.items():
        brightness_stats = brightness_data.get(track_id)
        comparison = tracklet.compute_best_detections_statistics(
            all_tracklets=list(tracklets_dict.values()),
            brightness_stats=brightness_stats
        )
        
        print(f"\nTrack {track_id}:")
        print(f"  Best detections ({comparison['best_detections']['count']}):")
        print(f"    Avg area: {comparison['best_detections']['average_area']:.1f}")
        print(f"    Avg score: {comparison['best_detections']['average_score']:.3f}")
        print(f"    Avg brightness: {comparison['best_detections']['average_brightness']:.1f}")
        print(f"    Avg contrast: {comparison['best_detections']['average_contrast']:.1f}")
        
        print(f"  All detections ({comparison['all_detections']['count']}):")
        print(f"    Avg area: {comparison['all_detections']['average_area']:.1f}")
        print(f"    Avg score: {comparison['all_detections']['average_score']:.3f}")
        print(f"    Avg brightness: {comparison['all_detections']['average_brightness']:.1f}")
        print(f"    Avg contrast: {comparison['all_detections']['average_contrast']:.1f}")
        
        # Calculate improvement ratios
        if comparison['all_detections']['average_area'] > 0:
            area_ratio = comparison['best_detections']['average_area'] / comparison['all_detections']['average_area']
            print(f"    Area improvement: {area_ratio:.2f}x")
        
        if comparison['all_detections']['average_brightness'] > 0:
            brightness_ratio = comparison['best_detections']['average_brightness'] / comparison['all_detections']['average_brightness']
            print(f"    Brightness ratio: {brightness_ratio:.2f}x")
    
    # Save results
    output_file = os.path.join(session_dir, "tracklet_brightness_contrast.json")
    import json
    with open(output_file, 'w') as f:
        json.dump(brightness_data, f, indent=2)
    
    print(f"\nBrightness/contrast statistics saved to: {output_file}")
    return brightness_data

# Example usage:
if __name__ == "__main__":
    stats = process_session_brightness_contrast("/home/aidev/workspace/reid/Thesis/reid-2024/trash/2025-05-14 16-36-09-96e885062e534a7e8433bca9b49d0f08")