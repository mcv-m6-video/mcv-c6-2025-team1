import cv2
import argparse
import numpy as np

from src.tracking.utils import iou, write_results_to_txt, read_detections_from_txt
from collections import defaultdict
from src.optical_flow.of import OpticalFlow


def track_overlap(detections: list, prev_frame: np.ndarray, curr_frame: np.ndarray, of_model: OpticalFlow) -> list:
    """Track objects across frames by assigning unique IDs based on the highest overlap (IoU) in consecutive frames. Also incorporates optical flow to improve tracking, especially in case of occlusion or overlap.

    Args:
        detections (list): List of detections in TXT format.
        prev_frame (np.ndarray): Previous frame matrix.
        curr_frame (np.ndarray): Current frame matrix.
        of_model (OpticalFlow): Optical flow model.

    Returns:
        list: List of tracking results with overlap.
    """
    track_id = 0  # Initialize track ID counter
    active_tracks = []  # List of active tracked objects
    track_eval_format = []  # Stores results in the required evaluation format
    frame_groups = defaultdict(list)  # Dictionary to group detections by frame ID

    # Step 1: Group detections by frame_id
    for detection in detections:
        frame_id = int(detection[0])
        frame_groups[frame_id].append(detection)

    # Step 2: Compute optical flow between previous and current frames
    optical_flow = of_model.compute_flow(prev_frame, curr_frame)

    # Step 3: Process detections frame by frame
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
            # For subsequent frames, find the best match using IoU and Optical Flow
            for detection in frame_detections:
                bbox_left, bbox_top, bbox_width, bbox_height, confidence_score = detection[1:]

                matched = False
                max_iou = 0
                best_match = None

                # Try to match detections based on optical flow predictions
                for track in active_tracks:
                    _, _, track_bbox_left, track_bbox_top, track_bbox_width, track_bbox_height, _ = track
                    
                    # Predict the new bounding box using optical flow
                    # Calculate the flow for the center of the bounding box
                    center_x = track_bbox_left + track_bbox_width / 2
                    center_y = track_bbox_top + track_bbox_height / 2

                    flow_u = optical_flow[int(center_y), int(center_x), 0]
                    flow_v = optical_flow[int(center_y), int(center_x), 1]

                    # Update the bounding box position based on flow
                    predicted_bbox = [
                        track_bbox_left + flow_u, 
                        track_bbox_top + flow_v, 
                        track_bbox_width, 
                        track_bbox_height
                    ]

                    # Compute IoU with the predicted bounding box
                    overlap = iou([bbox_left, bbox_top, bbox_width, bbox_height], predicted_bbox)

                    if overlap > 0.4 and overlap > max_iou:
                        max_iou = overlap
                        best_match = track

                # If a match is found using optical flow, update the track
                if best_match:
                    _, track_id, _, _, _, _, _ = best_match
                    if track_id not in assigned_ids:
                        assigned_ids.add(track_id)
                        active_tracks.remove(best_match)
                        active_tracks.append([frame_id, track_id, bbox_left, bbox_top, bbox_width, bbox_height, confidence_score])
                        track_eval_format.append(f"{frame_id}, {track_id}, {bbox_left}, {bbox_top}, {bbox_width}, {bbox_height}, {confidence_score}, -1, -1, -1\n")
                        matched = True

                # If no match is found, create a new unique track ID
                if not matched:
                    track_id += 1
                    
                    # Ensure the new ID is unique
                    while track_id in assigned_ids:
                        track_id += 1
                    assigned_ids.add(track_id)
                    
                    track_eval_format.append(f"{frame_id}, {track_id}, {bbox_left}, {bbox_top}, {bbox_width}, {bbox_height}, {confidence_score}, -1, -1, -1\n")
                    active_tracks.append([frame_id, track_id, bbox_left, bbox_top, bbox_width, bbox_height, confidence_score])
                    
    return track_eval_format


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Overlap algorithm for tracking with optical flow.")
    parser.add_argument("-d", "--detection_file_path", help="Path to the object detections TXT file.", required=True, type=str)
    parser.add_argument("-v", "--video_path", help="Path to the video file.", type=str, required=True)
    parser.add_argument("-o", "--output_path", help="Path to TXT file where the results will be stored", required=True, type=str)
    parser.add_argument("-m", "--of_model", help="Optical flow model to use.", required=False, default="rpknet", type=str, choices=["pyflow", "diclflow", "memflow", "rapidflow", "rpknet", "dip"])
    args = parser.parse_args()

    # Carregar les deteccions des del fitxer
    # detection_file_path = '/ghome/c5mcv01/mcv-c6-2025-team1/week2/src/tracking/detections_yolo.txt'
    detection_file_path = args.detection_file_path
    _, detections = read_detections_from_txt(detection_file_path)

    # Obrir el vídeo
    # video_path = '/ghome/c5mcv01/mcv-c6-2025-team1/data/AICity_data/train/S03/c010/vdo.avi'
    video_path = args.video_path
    cap = cv2.VideoCapture(video_path)

    # Comprovar si el vídeo s'ha obert correctament
    if not cap.isOpened():
        print("Error: No s'ha pogut obrir el vídeo.")
        raise ValueError

    # Inicialitzar variables
    prev_frame = None
    frame_id = 1
    track_eval_format = []

    # Agrupar les deteccions per fotograma per facilitar l'accés
    frame_groups = defaultdict(list)
    for detection in detections:
        frame_groups[detection[0]].append(detection)
        
    # Define optical flow model
    optical_flow = OpticalFlow(args.of_model)

    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break
        
        # Obtenir les deteccions per al fotograma actual (frame_id)
        print(f"Processant el fotograma: {frame_id}")
        if frame_id in frame_groups:
            frame_detections = frame_groups[frame_id]
        else:
            frame_detections = []

        # Si hi ha un fotograma anterior, cridem la funció de seguiment
        if prev_frame is not None:
            track_eval_format += track_overlap(frame_detections, prev_frame, curr_frame, optical_flow)

        # Actualitzar el fotograma anterior per al següent bucle
        prev_frame = curr_frame
        frame_id += 1
        
    cap.release()
    write_results_to_txt(track_eval_format, args.output_path)
