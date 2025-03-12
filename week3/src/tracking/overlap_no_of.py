import cv2
import argparse

from src.tracking.utils import write_results_to_txt, read_detections_from_txt
from src.tracking.overlap import calculate_iou
from collections import defaultdict


def track_overlap(frame_detections, current_frame_id, iou_threshold=0.4):
    """Track objects using only overlap (IOU)."""
    global active_tracks
    global next_track_id
    
    # Initialize empty dictionary for new tracks
    new_tracks = defaultdict(list)
    
    # If no active tracks, initialize new tracks for all detections
    if not active_tracks:
        for detection in frame_detections:
            _, left, top, width, height, conf = detection
            active_tracks[next_track_id] = {
                'bbox': (left, top, width, height),
                'last_seen': current_frame_id
            }
            new_tracks[current_frame_id].append([current_frame_id, next_track_id, left, top, width, height, conf])
            next_track_id += 1
        return new_tracks

    # Calculate IOUs between all active tracks and current detections
    unmatched_detections = frame_detections.copy()
    matched_track_ids = set()

    for track_id, track_info in active_tracks.items():
        track_bbox = track_info['bbox']
        best_iou = iou_threshold
        best_detection = None
        best_detection_idx = None

        for idx, detection in enumerate(unmatched_detections):
            _, left, top, width, height, conf = detection
            detection_bbox = (left, top, width, height)
            iou = calculate_iou(track_bbox, detection_bbox)

            if iou > best_iou:
                best_iou = iou
                best_detection = detection
                best_detection_idx = idx

        if best_detection is not None:
            # Update track with matched detection
            _, left, top, width, height, conf = best_detection
            active_tracks[track_id] = {
                'bbox': (left, top, width, height),
                'last_seen': current_frame_id
            }
            new_tracks[current_frame_id].append([current_frame_id, track_id, left, top, width, height, conf])
            matched_track_ids.add(track_id)
            unmatched_detections.pop(best_detection_idx)

    # Initialize new tracks for unmatched detections
    for detection in unmatched_detections:
        _, left, top, width, height, conf = detection
        active_tracks[next_track_id] = {
            'bbox': (left, top, width, height),
            'last_seen': current_frame_id
        }
        new_tracks[current_frame_id].append([current_frame_id, next_track_id, left, top, width, height, conf])
        next_track_id += 1

    # Remove tracks that haven't been seen recently
    track_ids = list(active_tracks.keys())
    for track_id in track_ids:
        if active_tracks[track_id]['last_seen'] < current_frame_id - 1:
            del active_tracks[track_id]

    return new_tracks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple overlap tracking algorithm")
    parser.add_argument("-d", "--detection_file_path", required=True, type=str, help="Path to the object detections TXT file")
    parser.add_argument("-v", "--video_path", required=True, type=str, help="Path to the video file")
    parser.add_argument("-o", "--output_path", required=True, type=str, help="Path to output TXT file")
    parser.add_argument("-ov", "--output_video_path", required=True, type=str, help="Path to the output video")
    parser.add_argument("--iou_threshold", default=0.4, type=float, help="Threshold for IoU matching")
    args = parser.parse_args()

    # Initialize global variables
    active_tracks = {}
    next_track_id = 0

    # Process video frames and detections
    cap = cv2.VideoCapture(args.video_path)
    output_video = cv2.VideoWriter(
        args.output_video_path, 
        cv2.VideoWriter_fourcc(*'XVID'), 
        cap.get(cv2.CAP_PROP_FPS), 
        (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    )
    
    # Get all frame detections
    _, detections = read_detections_from_txt(args.detection_file_path)
    
    # Group detections by frame for easier access
    frame_groups = defaultdict(list)
    for detection in detections:
        frame_groups[detection[0]].append(detection)

    frame_id = 1
    all_tracks = defaultdict(list)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        print(f"Processing frame: {frame_id}")
        # Read detections for current frame
        if frame_id in frame_groups:
            frame_detections = frame_groups[frame_id]
        else:
            frame_detections = []
        
        # Perform tracking
        new_tracks = track_overlap(frame_detections, frame_id, args.iou_threshold)
        
        # Update all_tracks with new tracks
        for track in new_tracks[frame_id]:
            track_str = f"{int(track[0])},{int(track[1])},{int(track[2])},{int(track[3])},{int(track[4])},{int(track[5])},-1, -1, -1, -1\n"
            all_tracks[frame_id].append(track_str)

        # Visualize tracks
        frame_copy = frame.copy()
        for track in new_tracks[frame_id]:
            _, track_id, left, top, width, height, _ = track
            x1, y1 = int(left), int(top)
            x2, y2 = int(left + width), int(top + height)
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame_copy, f"ID: {track_id}", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        output_video.write(frame_copy)
        frame_id += 1

    # Clean up
    cap.release()
    output_video.release()
    
    # Write results to output file
    flattened_tracks = []
    for frame_id in sorted(all_tracks.keys()):
        flattened_tracks.extend(all_tracks[frame_id])
    write_results_to_txt(flattened_tracks, args.output_path)
    print(f"Tracking results saved to {args.output_path}")
    print(f"Annotated video saved to {args.output_video_path}")