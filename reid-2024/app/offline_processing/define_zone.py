import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt

def define_reliable_uncertain_zones(frame, visualize=True):
    """
    Define reliable and uncertain zones for people tracking.
    
    Args:
        frame: Input frame or frame dimensions as (height, width)
        visualize: Whether to visualize the zones
        
    Returns:
        reliable_zone: List of polygon coordinates for reliable zone
        uncertain_zones: List of polygons for uncertain zones
    """
    if isinstance(frame, np.ndarray):
        height, width = frame.shape[:2]
    else:
        height, width = frame
        
    # Define margins for reliable zone (in pixels or as ratio of frame size)
    margin_x = int(width * 0.1)  # 10% margin from sides
    margin_y_near = int(height * 0.1)  # 10% margin from bottom (near camera)
    margin_y_far = int(height * 0.2)   # 20% margin from top (far end of hallway)
    
    # Define reliable zone (central area)
    reliable_zone = np.array([
        [margin_x, margin_y_near],                   # Bottom left
        [width - margin_x, margin_y_near],           # Bottom right
        [width - margin_x, height - margin_y_far],   # Top right
        [margin_x, height - margin_y_far]            # Top left
    ], dtype=np.int32)

    left_edge = np.array([
        [0, 0],
        [margin_x, 0],
        [margin_x, height],
        [0, height]
    ], dtype=np.int32)
    
    # Right edge
    right_edge = np.array([
        [width - margin_x, 0],
        [width, 0],
        [width, height],
        [width - margin_x, height]
    ], dtype=np.int32)
    
    # Near edge (bottom)
    near_edge = np.array([
        [margin_x, 0],
        [width - margin_x, 0],
        [width - margin_x, margin_y_near],
        [margin_x, margin_y_near]
    ], dtype=np.int32)
    
    # Far edge (top)
    far_edge = np.array([
        [margin_x, height - margin_y_far],
        [width - margin_x, height - margin_y_far],
        [width - margin_x, height],
        [margin_x, height]
    ], dtype=np.int32)
    
    uncertain_zones = [left_edge, right_edge, near_edge, far_edge]
    
    if visualize and isinstance(frame, np.ndarray):
        # Create a copy of the frame for visualization
        vis_frame = frame.copy()
        
        # Draw reliable zone (green, semi-transparent)
        overlay = vis_frame.copy()
        cv2.fillPoly(overlay, [reliable_zone], (0, 255, 0, 128))
        cv2.addWeighted(overlay, 0.3, vis_frame, 0.7, 0, vis_frame)
        cv2.polylines(vis_frame, [reliable_zone], True, (0, 255, 0), 2)
        cv2.putText(vis_frame, "Reliable Zone", (reliable_zone[0][0] + 10, reliable_zone[0][1] + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw uncertain zones (red, semi-transparent)
        overlay = vis_frame.copy()
        for i, zone in enumerate(uncertain_zones):
            cv2.fillPoly(overlay, [zone], (0, 0, 255, 128))
            cv2.addWeighted(overlay, 0.3, vis_frame, 0.7, 0, vis_frame)
            cv2.polylines(vis_frame, [zone], True, (0, 0, 255), 2)
            
        # Label uncertain zones
        zone_names = ["Left Edge", "Right Edge", "Near Edge", "Far Edge"]
        positions = [
            (10, height // 2),
            (width - 150, height // 2),
            (width // 2, 30),
            (width // 2, height - 50)
        ]
        
        for name, pos in zip(zone_names, positions):
            cv2.putText(vis_frame, f"Uncertain: {name}", pos, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
        # Display the result
        cv2.imshow("Zones Visualization", vis_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Save the visualization
        cv2.imwrite("zones_visualization.jpg", vis_frame)
        print("Visualization saved as zones_visualization.jpg")
    
    return reliable_zone, uncertain_zones

def is_in_reliable_zone(bbox, reliable_zone):
    """
    Check if a bounding box is in the reliable zone.
    
    Args:
        bbox: Bounding box [x1, y1, x2, y2] or [x, y, w, h]
        reliable_zone: Polygon defining reliable zone
        
    Returns:
        True if bbox center is in reliable zone, False otherwise
    """
    # Convert bbox to center point
    if len(bbox) == 4:
        if bbox[2] < bbox[0] or bbox[3] < bbox[1]:  # x2 < x1 or y2 < y1 means it's in [x,y,w,h] format
            x, y, w, h = bbox
            center_x = x + w/2
            center_y = y + h/2
        else:  # It's in [x1,y1,x2,y2] format
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
    
    # Create a point
    point = (center_x, center_y)
    
    # Check if point is inside polygon
    return cv2.pointPolygonTest(reliable_zone, point, False) >= 0

def filter_tracklets_by_zone(tracklets, reliable_zone):
    """
    Filter tracklets into reliable and uncertain groups.
    
    Args:
        tracklets: List of tracklet objects with bboxes property
        reliable_zone: Polygon defining reliable zone
        
    Returns:
        reliable_tracklets: List of tracklets primarily in reliable zone
        uncertain_tracklets: List of tracklets primarily in uncertain zone
    """
    reliable_tracklets = []
    uncertain_tracklets = []
    
    for tracklet in tracklets:
        # Count how many detections are in reliable zone
        reliable_count = 0
        total_boxes = len(tracklet.bboxes)
        
        for bbox in tracklet.bboxes:
            if is_in_reliable_zone(bbox, reliable_zone):
                reliable_count += 1
        
        # If more than 70% of detections are in reliable zone, consider tracklet reliable
        if reliable_count / total_boxes > 0.7:
            reliable_tracklets.append(tracklet)
        else:
            uncertain_tracklets.append(tracklet)
    
    return reliable_tracklets, uncertain_tracklets

# Function to integrate into your tracking system
def calculate_reliable_features(tracklets, frame_shape):
    """
    Calculate reliable features for tracklets by filtering out detections in uncertain zones.
    
    Args:
        tracklets: List of tracklet objects
        frame_shape: Tuple of (height, width) for frame dimensions
        
    Returns:
        processed_tracklets: Tracklets with features calculated only from reliable detections
    """
    reliable_zone, _ = define_reliable_uncertain_zones(frame_shape, visualize=False)
    processed_tracklets = []
    
    for tracklet in tracklets:
        # Create a copy to avoid modifying original
        new_tracklet = copy.deepcopy(tracklet)
        
        # Get indices of bboxes in reliable zone
        reliable_indices = []
        for i, bbox in enumerate(tracklet.bboxes):
            if is_in_reliable_zone(bbox, reliable_zone):
                reliable_indices.append(i)
        
        # If we have enough reliable detections, use only those
        if len(reliable_indices) > max(3, len(tracklet.bboxes) * 0.3):  # At least 3 or 30% of detections
            # Extract reliable features, bboxes, etc.
            new_tracklet.features = [tracklet.features[i] for i in reliable_indices]
            new_tracklet.bboxes = [tracklet.bboxes[i] for i in reliable_indices]
            new_tracklet.frames = [tracklet.frames[i] for i in reliable_indices]
            if hasattr(tracklet, 'scores'):
                new_tracklet.scores = [tracklet.scores[i] for i in reliable_indices]
        
        processed_tracklets.append(new_tracklet)
    
    return processed_tracklets

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Define and visualize reliable/uncertain zones")
    parser.add_argument("--image", type=str, required=True, help="Path to sample image")
    args = parser.parse_args()
    
    