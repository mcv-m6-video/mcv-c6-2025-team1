import cv2
import argparse

from src.tracking.utils import write_results_to_txt, read_detections_from_txt
from collections import defaultdict
from src.optical_flow.of import OpticalFlow


def track_overlap(frame_detections, prev_frame, curr_frame, optical_flow, current_frame_id, iou_threshold: float = 0.4):
    """
    Track objects using overlap (IOU) and optical flow.
    
    Args:
        frame_detections (list): List of detections for the current frame [frame_id, -1, left, top, width, height, conf]
        prev_frame (np.ndarray): Previous frame
        curr_frame (np.ndarray): Current frame
        optical_flow (OpticalFlow): Optical flow processor
        current_frame_id (int): Current frame ID
        iou_threshold (float): Threshold for IoU of predicted and detected boxes.
        
    Returns:
        list: Updated tracks in evaluation format with newlines
    """
    global active_tracks, next_track_id
    
    # Initialize variables if not yet initialized
    if 'active_tracks' not in globals():
        global active_tracks
        active_tracks = {}  # {track_id: {'bbox': [left, top, width, height], 'last_seen': frame_id}}
    
    if 'next_track_id' not in globals():
        global next_track_id
        next_track_id = 1
        
    # Compute optical flow between previous and current frame
    flow = optical_flow.compute_flow(prev_frame, curr_frame)
    
    # Update active tracks using optical flow (predict new positions)
    predicted_tracks = {}
    for track_id, track_info in active_tracks.items():
        bbox = track_info['bbox']
        left, top, width, height = bbox
        
        # Calculate center of the bounding box
        center_x = left + width / 2
        center_y = top + height / 2
        
        # Get optical flow at the center position (rounding to integers for indexing)
        center_x_int, center_y_int = int(center_x), int(center_y)
        
        # Ensure the coordinates are within image bounds
        h, w = flow.shape[:2]
        center_x_int = max(0, min(center_x_int, w - 1))
        center_y_int = max(0, min(center_y_int, h - 1))
        
        # Get flow at the center
        flow_u, flow_v = flow[center_y_int, center_x_int]
        
        # Update bounding box position based on flow
        new_left = left + flow_u
        new_top = top + flow_v
        
        # Ensure the box stays within frame boundaries
        new_left = max(0, new_left)
        new_top = max(0, new_top)
        
        # Store predicted bbox
        predicted_tracks[track_id] = {
            'bbox': [new_left, new_top, width, height],
            'last_seen': track_info['last_seen']
        }
    
    # Prepare current detections for matching
    current_detections = []
    for det in frame_detections:
        _, left, top, width, height, conf = det
        current_detections.append({
            'bbox': [left, top, width, height],
            'conf': conf
        })
    
    # Match detections to predicted tracks using IOU
    matched_track_ids = set()
    matched_detection_indices = set()
    
    # Calculate IOUs between all predicted tracks and current detections
    iou_matrix = []
    for track_id, track_info in predicted_tracks.items():
        track_bbox = track_info['bbox']
        track_iou_row = []
        
        for i, det in enumerate(current_detections):
            det_bbox = det['bbox']
            iou = calculate_iou(track_bbox, det_bbox)
            track_iou_row.append((iou, i, track_id))
        
        if track_iou_row:  # Only if we have detections
            iou_matrix.extend(track_iou_row)
    
    # Sort IOUs in descending order
    iou_matrix.sort(reverse=True)
    
    # Assign detections to tracks using greedy algorithm
    for iou, det_idx, track_id in iou_matrix:
        if iou < iou_threshold:
            continue
            
        if det_idx not in matched_detection_indices and track_id not in matched_track_ids:
            matched_detection_indices.add(det_idx)
            matched_track_ids.add(track_id)
            
            # Update the track with detection
            det = current_detections[det_idx]
            active_tracks[track_id] = {
                'bbox': det['bbox'],
                'last_seen': current_frame_id
            }
    
    # Create new tracks for unmatched detections
    for i, det in enumerate(current_detections):
        if i not in matched_detection_indices:
            active_tracks[next_track_id] = {
                'bbox': det['bbox'],
                'last_seen': current_frame_id
            }
            next_track_id += 1
    
    # Remove tracks that haven't been seen for a while
    max_age = 0  # TODO: MAX_AGE param
    track_ids_to_remove = []
    
    for track_id, track_info in active_tracks.items():
        if current_frame_id - track_info['last_seen'] > max_age:
            track_ids_to_remove.append(track_id)
    
    for track_id in track_ids_to_remove:
        del active_tracks[track_id]
    
    # Convert tracks to evaluation format: frame_id, track_id, left, top, width, height, -1, -1, -1, -1
    result_tracks = []
    for track_id, track_info in active_tracks.items():
        left, top, width, height = track_info['bbox']
        
        # Only add tracks updated in the current frame
        if track_info['last_seen'] == current_frame_id:
            # Convert to integers and create properly formatted string
            int_left = int(round(left))
            int_top = int(round(top))
            int_width = int(round(width))
            int_height = int(round(height))
            
            # Format with spaces after commas and add a newline at the end
            # This is crucial for writelines() to work correctly
            track_str = f"{current_frame_id}, {track_id}, {int_left}, {int_top}, {int_width}, {int_height}, -1, -1, -1, -1\n"
            result_tracks.append(track_str)
    
    print(result_tracks)
    return result_tracks


def calculate_iou(bbox1, bbox2):
    """
    Calculate Intersection over Union (IOU) between two bounding boxes.
    
    Args:
        bbox1 (list): First bounding box [left, top, width, height]
        bbox2 (list): Second bounding box [left, top, width, height]
        
    Returns:
        float: IOU value
    """
    left1, top1, width1, height1 = bbox1
    left2, top2, width2, height2 = bbox2
    
    # Calculate coordinates of the intersection rectangle
    x_left = max(left1, left2)
    y_top = max(top1, top2)
    x_right = min(left1 + width1, left2 + width2)
    y_bottom = min(top1 + height1, top2 + height2)
    
    # Check if there is an intersection
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    # Calculate intersection area
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate union area
    bbox1_area = width1 * height1
    bbox2_area = width2 * height2
    union_area = bbox1_area + bbox2_area - intersection_area
    
    # Calculate IOU
    iou = intersection_area / union_area
    
    return iou


def initialize_tracks_from_detections(frame_detections, current_frame_id):
    """
    Initialize tracks from detections in the first frame.
    
    Args:
        frame_detections (list): List of detections for the first frame
        current_frame_id (int): Current frame ID (should be 1)
        
    Returns:
        list: New tracks in evaluation format with newlines
    """
    global active_tracks, next_track_id
    
    # Initialize variables if not yet initialized
    if 'active_tracks' not in globals():
        global active_tracks
        active_tracks = {}  # {track_id: {'bbox': [left, top, width, height], 'last_seen': frame_id}}
    
    if 'next_track_id' not in globals():
        global next_track_id
        next_track_id = 1
    
    # Create tracks from detections
    result_tracks = []
    for det in frame_detections:
        _, left, top, width, height, conf = det
        
        # Create a new track
        active_tracks[next_track_id] = {
            'bbox': [left, top, width, height],
            'last_seen': current_frame_id
        }
        
        # Format track for output
        track_str = f"{current_frame_id}, {next_track_id}, {int(left)}, {int(top)}, {int(width)}, {int(height)}, -1, -1, -1, -1\n"
        result_tracks.append(track_str)
        
        next_track_id += 1
    
    return result_tracks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Overlap algorithm for tracking with optical flow.")
    parser.add_argument("-d", "--detection_file_path", help="Path to the object detections TXT file.", required=True, type=str)
    parser.add_argument("-v", "--video_path", help="Path to the video file.", type=str, required=True)
    parser.add_argument("-o", "--output_path", help="Path to TXT file where the results will be stored", required=True, type=str)
    parser.add_argument("-ov", "--output_video_path", help="Path to the output video.", required=True, type=str)
    parser.add_argument("-m", "--of_model", help="Optical flow model to use.", required=False, default="rpknet", type=str, choices=["pyflow", "diclflow", "memflow", "rapidflow", "rpknet", "dip"])
    parser.add_argument("--iou_threshold", help="Threshold for IoU of predicted and detected boxes.", required=False, default=0.4, type=float)
    args = parser.parse_args()

    # Load detections from file
    _, detections = read_detections_from_txt(args.detection_file_path)

    # Open the video
    video_path = args.video_path
    cap = cv2.VideoCapture(video_path)

    # Check if the video was opened correctly
    if not cap.isOpened():
        print("Error: Could not open the video.")
        raise ValueError

    # Get video information
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(args.output_video_path, fourcc, fps, (frame_width, frame_height))

    # Initialize variables
    prev_frame = None
    frame_id = 1
    track_eval_format = []
    active_tracks = {}  # Keep track of active tracks
    next_track_id = 1   # Initialize track ID counter

    # Group detections by frame for easier access
    frame_groups = defaultdict(list)
    for detection in detections:
        frame_groups[detection[0]].append(detection)
        
    # Define optical flow model
    optical_flow = OpticalFlow(args.of_model)

    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break
        
        # Get detections for the current frame
        print(f"Processing frame: {frame_id}")
        curr_frame_copy = curr_frame.copy()  # Create a copy for drawing
        
        if frame_id in frame_groups:
            frame_detections = frame_groups[frame_id]
        else:
            frame_detections = []

        # Handle first frame separately (no optical flow needed)
        if prev_frame is None:
            # For the first frame, simply initialize tracks from detections
            new_tracks = initialize_tracks_from_detections(frame_detections, frame_id)
        else:
            # For subsequent frames, use optical flow tracking
            new_tracks = track_overlap(frame_detections, prev_frame, curr_frame, optical_flow, frame_id, args.iou_threshold)
        
        track_eval_format += new_tracks
        
        # Update active tracks for visualization
        for track in new_tracks:
            parts = track.strip().split(',')
            track_frame_id = int(parts[0])
            track_id = int(parts[1])
            bbox_left = float(parts[2])
            bbox_top = float(parts[3])
            bbox_width = float(parts[4])
            bbox_height = float(parts[5])
            
            if track_frame_id == frame_id:
                active_tracks[track_id] = {
                    'bbox': [bbox_left, bbox_top, bbox_width, bbox_height],
                    'last_seen': frame_id
                }
        
        # Draw current active tracks on the frame
        for track_id, track_info in active_tracks.items():
            bbox_left, bbox_top, bbox_width, bbox_height = track_info['bbox']
            x1, y1 = int(bbox_left), int(bbox_top)
            x2, y2 = int(bbox_left + bbox_width), int(bbox_top + bbox_height)
            
            # Draw bounding box
            cv2.rectangle(curr_frame_copy, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # Draw track ID
            cv2.putText(curr_frame_copy, f"ID: {track_id}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Write the frame to the output video
        out.write(curr_frame_copy)
        
        # Update the previous frame for the next loop
        prev_frame = curr_frame
        frame_id += 1
        
    # Release video capture and writer
    cap.release()
    out.release()
    
    # Write tracking results to file
    write_results_to_txt(track_eval_format, args.output_path)
    print(f"Tracking results saved to {args.output_path}")
    print(f"Annotated video saved to {args.output_video_path}")
