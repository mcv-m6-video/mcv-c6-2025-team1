from utils import iou
from collections import defaultdict
import cv2
import numpy as np
import ptlflow
from ptlflow.utils.io_adapter import IOAdapter
from collections import defaultdict


def compute_optical_flow(img1, img2):
    """
    Compute optical flow using RAFT (Recurrent All-Pairs Field Transforms) between two images.

    Args:
        img1 (np.array): The first image (prev_frame).
        img2 (np.array): The second image (curr_frame).

    Returns:
        np.array: Optical flow in the form of (u, v) components stacked together.
    """
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    model_ptlflow = ptlflow.get_model('rpknet', ckpt_path='things')
    io_adapter = IOAdapter(model_ptlflow, img1_rgb.shape[:2])
    inputs = io_adapter.prepare_inputs([img1_rgb, img2_rgb])

    predictions = model_ptlflow(inputs)
    flow = predictions['flows'][0, 0]  # Remove batch and sequence dimensions
    flow = flow.permute(1, 2, 0)  # Convert from CHW to HWC format

    u = flow[:, :, 0].detach().numpy()  # Horizontal flow component
    v = flow[:, :, 1].detach().numpy()  # Vertical flow component

    return np.dstack((u, v))


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


def track_overlap(detections, prev_frame, curr_frame):
    """
    Track objects across frames by assigning unique IDs based on the highest overlap (IoU) in consecutive frames.
    Also incorporates optical flow to improve tracking, especially in case of occlusion or overlap.
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
    optical_flow = compute_optical_flow(prev_frame, curr_frame)

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
                for i, track in enumerate(active_tracks):
                    track_frame_id, existing_track_id, track_bbox_left, track_bbox_top, track_bbox_width, track_bbox_height, track_confidence = track
                    
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



# Carregar les deteccions des del fitxer
detection_file_path = '/ghome/c5mcv01/mcv-c6-2025-team1/week2/src/tracking/detections_yolo.txt'
detections = read_detections_from_txt(detection_file_path)

# Obrir el vídeo
video_path = '/ghome/c5mcv01/mcv-c6-2025-team1/data/AICity_data/train/S03/c010/vdo.avi'
cap = cv2.VideoCapture(video_path)

# Comprovar si el vídeo s'ha obert correctament
if not cap.isOpened():
    print("Error: No s'ha pogut obrir el vídeo.")
    exit()

# Inicialitzar variables
prev_frame = None
frame_id = 1
track_eval_format = []

# Agrupar les deteccions per fotograma per facilitar l'accés
frame_groups = defaultdict(list)
for detection in detections:
    frame_groups[detection[0]].append(detection)

while True:
    # Llegir el fotograma següent del vídeo
    ret, curr_frame = cap.read()
    
    if not ret:
        break  # Si no hi ha més fotogrames, sortim del bucle
    
    print(f"Processant el fotograma: {frame_id}")
    # Obtenir les deteccions per al fotograma actual (frame_id)
    if frame_id in frame_groups:
        frame_detections = frame_groups[frame_id]
    else:
        frame_detections = []

    # Si hi ha un fotograma anterior, cridem la funció de seguiment
    if prev_frame is not None:
        # Cridar la funció de seguiment amb optical flow i les deteccions del fotograma actual
        track_eval_format += track_overlap(frame_detections, prev_frame, curr_frame)

    # Actualitzar el fotograma anterior per al següent bucle
    prev_frame = curr_frame
    frame_id += 1

# Alliberar el vídeo
cap.release()




write_results_to_txt(track_eval_format, '/ghome/c5mcv01/mcv-c6-2025-team1/week3/src/tracking/s03_of.txt')
#write_results_to_txt(track_eval_format, '/ghome/c3mcv02/mcv-c6-2025-team1/week2/src/tracking/TrackEval/data/trackers/mot_challenge/week2-train/overlap/data/s03.txt')








                
                