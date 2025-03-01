from utils import iou
from collections import defaultdict
import cv2
import numpy as np
from sort import Sort

def read_detections_from_txt(file_path):
    detections = {}
    detections_vect = []
    with open(file_path, 'r') as file:
        for line in file:
            data = line.strip().split(',')
            frame_id = int(data[0])
            bbox_left = float(data[2])
            bbox_top = float(data[3])
            bbox_width = float(data[4])
            bbox_height = float(data[5])
            confidence_score = float(data[6])

            # Store in dictionary with frame_id as key
            if frame_id not in detections:
                detections[frame_id] = []
            
            detections[frame_id].append({
                "bb_left": bbox_left,
                "bb_top": bbox_top,
                "bb_w": bbox_width,
                "bb_h": bbox_height,
                "score": confidence_score
            })

            detections_vect.append([frame_id, bbox_left, bbox_top,bbox_width,bbox_height, confidence_score])

    return detections, detections_vect


def write_results_to_txt(track_eval_format, output_path):
    with open(output_path, 'w') as file:
        file.writelines(track_eval_format)
    print(f"Results written to {output_path}")





detections, detections_vect = read_detections_from_txt('/ghome/c3mcv02/mcv-c6-2025-team1/data/AICity_data/train/S03/c010/det/det_yolo3.txt')  
#print(f"Detections: {detections}")

# Create instance of the SORT tracker (default params: max_age=1, min_hits=3, iou_threshold=0.3)
mot_tracker = Sort() 

# Open the video
cap = cv2.VideoCapture("/ghome/c3mcv02/mcv-c6-2025-team1/data/AICity_data/train/S03/c010/vdo.avi")

# Get video information (frame size, fps, etc.)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Prepare the output video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter("/ghome/c3mcv02/mcv-c6-2025-team1/week2/src/tracking/kf.avi", fourcc, fps, (frame_width, frame_height))

# Process each frame of the video
track_eval_format = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Get the current frame number
    frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    print("-" * 50)
    print(f"Processing frame {frame_number}")
    
    actual_bb = [detection[1:5] for detection in detections_vect if detection[0] == frame_number]  
    actual_bb = np.array(actual_bb)
    #print(f"Actual BB frame {frame_number}: {actual_bb}")
    

    if len(actual_bb) > 0:
        # Convert [x1, y1, w, h] -> [x1, y1, x2, y2] format for the tracker
        actual_bb[:, 2] += actual_bb[:, 0]  # x2 = x1 + w
        actual_bb[:, 3] += actual_bb[:, 1]  # y2 = y1 + h

        # Default params: dets --> np.array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        # Returns a similar np.array, where the last column is the object ID
        tracked_cars = mot_tracker.update(actual_bb)

        print(f"Tracking info for frame {frame_number}: {tracked_cars}")

    else:
        # Must be called once per frame
        tracked_cars = mot_tracker.update(np.empty((0, 5)))

    #track_eval_format.append(f"{frame_number}, {track_id}, {bbox_left}, {bbox_top}, {bbox_width}, {bbox_height}, {confidence_score}, -1, -1, -1\n")
    # Draw tracked 2DBB
    for obj in tracked_cars:
            x1, y1, x2, y2, track_id = map(int, obj)
            # Draw rectangle (red)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            # Add track ID text
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Store the information about the tracked objects
            track_eval_format.append(f"{frame_number}, {track_id}, {x1}, {y1}, {x2-x1}, {y2-y1}, -1, -1, -1, -1\n")


    # Write the frame with bounding boxes and track IDs to the output video
    out.write(frame)

# Release resources
cap.release()
out.release()

write_results_to_txt(track_eval_format, '/ghome/c3mcv02/mcv-c6-2025-team1/week2/src/tracking/TrackEval/data/trackers/mot_challenge/week2-train/overlap/data/s03.txt')

print(f"Annotated video with bounding boxes and track IDs saved")












                
                