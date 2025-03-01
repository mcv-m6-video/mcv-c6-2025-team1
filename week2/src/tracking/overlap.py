from utils import iou
from collections import defaultdict
def read_detections_from_txt(file_path):
    frames = []
    with open(file_path, 'r') as file:
        for line in file:
            data = line.strip().split(',')
            frame_id = int(data[0])
            bbox_left = float(data[2])
            bbox_top = float(data[3])
            bbox_width = float(data[4])
            bbox_height = float(data[5]) 
            confidence_score = float(data[6])           
            
            frames.append([frame_id, bbox_left, bbox_top,bbox_width,bbox_height, confidence_score])
    return frames


def track_overlap(detections):
    """
    Track objects across frames by assigning unique IDs based on the highest overlap (IoU) in consecutive frames.
    Ensures that each object in a frame has a unique track ID to prevent errors in MOTChallenge evaluation.
    """
    track_id = 0  # Initialize track ID counter
    active_tracks = []  # List of active tracked objects
    track_eval_format = []  # Stores results in the required evaluation format
    frame_groups = defaultdict(list)  # Dictionary to group detections by frame ID

    # Step 1: Group detections by frame_id
    for detection in detections:
        frame_id = int(detection[0])
        frame_groups[frame_id].append(detection)

    # Step 2: Process detections frame by frame
    for frame_id in sorted(frame_groups.keys()):
        frame_detections = frame_groups[frame_id]
        assigned_ids = set()  # Tracks assigned IDs to avoid duplicates in the same frame
        
        if not active_tracks:
            # If it's the first frame, assign unique track IDs to all detections
            for detection in frame_detections:
                bbox_left, bbox_top, bbox_width, bbox_height, confidence_score = detection[1:]
                track_id += 1  
                assigned_ids.add(track_id)  # Store assigned ID
                track_eval_format.append(f"{frame_id}, {track_id}, {bbox_left}, {bbox_top}, {bbox_width}, {bbox_height}, {confidence_score}, -1, -1, -1\n")
                active_tracks.append([frame_id, track_id, bbox_left, bbox_top, bbox_width, bbox_height, confidence_score])
        else:
            # For subsequent frames, find the best match using IoU
            for detection in frame_detections:
                bbox_left, bbox_top, bbox_width, bbox_height, confidence_score = detection[1:]

                matched = False
                max_iou = 0
                best_match = None

                # Find the best matching track based on IoU
                for track in active_tracks:
                    track_frame_id, existing_track_id, track_bbox_left, track_bbox_top, track_bbox_width, track_bbox_height, track_confidence = track
                    overlap = iou([bbox_left, bbox_top, bbox_width, bbox_height], [track_bbox_left, track_bbox_top, track_bbox_width, track_bbox_height])

                    if overlap > 0.5 and overlap > max_iou:
                        max_iou = overlap
                        best_match = track

                # If a match is found, assign the existing track ID
                if best_match:
                    track_frame_id, track_id, _, _, _, _, _ = best_match
                    if track_id not in assigned_ids:  # Prevent duplicate IDs in the same frame
                        assigned_ids.add(track_id)
                        active_tracks.remove(best_match)  # Remove outdated track
                        active_tracks.append([frame_id, track_id, bbox_left, bbox_top, bbox_width, bbox_height, confidence_score])  # Update track
                        track_eval_format.append(f"{frame_id}, {track_id}, {bbox_left}, {bbox_top}, {bbox_width}, {bbox_height}, {confidence_score}, -1, -1, -1\n")
                        matched = True

                # If no match is found, create a new unique track ID
                if not matched:
                    track_id += 1
                    while track_id in assigned_ids:  # Ensure the new ID is unique
                        track_id += 1
                    assigned_ids.add(track_id)
                    track_eval_format.append(f"{frame_id}, {track_id}, {bbox_left}, {bbox_top}, {bbox_width}, {bbox_height}, {confidence_score}, -1, -1, -1\n")
                    active_tracks.append([frame_id, track_id, bbox_left, bbox_top, bbox_width, bbox_height, confidence_score])

    return track_eval_format

            
        
            
def write_results_to_txt(track_eval_format, output_path):
    with open(output_path, 'w') as file:
        file.writelines(track_eval_format)
    print(f"Results written to {output_path}")





frames = read_detections_from_txt('/ghome/c3mcv02/mcv-c6-2025-team1/data/AICity_data/train/S03/c010/det/det_yolo3.txt')  

track_eval_format = track_overlap(frames)

write_results_to_txt(track_eval_format, '/ghome/c3mcv02/mcv-c6-2025-team1/week2/src/tracking/TrackEval/data/trackers/mot_challenge/week2-train/overlap/data/s03.txt')









                
                