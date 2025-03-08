from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import numpy as np
import ptlflow
from ptlflow.utils.io_adapter import IOAdapter

def compute_optical_flow(img1, img2):
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    model_ptlflow = ptlflow.get_model('rapidflow', ckpt_path='things')
    io_adapter = IOAdapter(model_ptlflow, img1_rgb.shape[:2])
    inputs = io_adapter.prepare_inputs([img1_rgb, img2_rgb])

    predictions = model_ptlflow(inputs)
    flow = predictions['flows'][0, 0].permute(1, 2, 0)

    u = flow[:, :, 0].detach().numpy()
    v = flow[:, :, 1].detach().numpy()

    return np.dstack((u, v))

def read_detections_from_txt(file_path):
    detections = []
    with open(file_path, 'r') as file:
        for line in file:
            data = line.strip().split(',')
            frame_id = int(data[0])
            x1 = float(data[2])
            y1 = float(data[3])
            w = float(data[4])
            h = float(data[5])
            score = float(data[6])
            detections.append([frame_id, x1, y1, x1 + w, y1 + h, score])
    return detections

def write_results_to_txt(track_eval_format, output_path):
    with open(output_path, 'w') as file:
        file.writelines(track_eval_format)
    print(f"Results written to {output_path}")

detections_vect = read_detections_from_txt('/ghome/c5mcv01/mcv-c6-2025-team1/week3/src/tracking/detections_yolo.txt')  
tracker = DeepSort(max_age=30, n_init=3, max_cosine_distance=0.3)

cap = cv2.VideoCapture("/ghome/c5mcv01/mcv-c6-2025-team1/data/AICity_data/train/S03/c010/vdo.avi")
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('kf_deepsort.avi', fourcc, fps, (frame_width, frame_height))

prev_frame = None
track_eval_format = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    actual_bb = [[float(d[1]), float(d[2]), float(d[1] + d[3]), float(d[2] + d[4]), float(d[5])] for d in detections_vect if d[0] == frame_number]

    print(actual_bb)

    if prev_frame is not None:
        flow = compute_optical_flow(prev_frame, frame)
        for i, bbox in enumerate(actual_bb):
            x1, y1, w, h = bbox
            flow_x = flow[int(y1), int(x1)][0]  # Horizontal flow component at top-left corner
            flow_y = flow[int(y1), int(x1)][1]  # Vertical flow component at top-left corner
            actual_bb[i, 0] += flow_x  # Update x-coordinate based on flow
            actual_bb[i, 1] += flow_y  # Update y-coordinate based on flow

    if len(actual_bb) > 0:
        tracked_objects = tracker.update_tracks(actual_bb, frame=frame)
    else:
        tracked_objects = tracker.update_tracks((np.empty((0, 5))), frame=frame)

    for obj in tracked_objects:
        track_id, x1, y1, x2, y2 = obj
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (int(x1), int(y1) - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        track_eval_format.append(f"{frame_number}, {track_id}, {int(x1)}, {int(y1)}, {int(x2-x1)}, {int(y2-y1)}, -1, -1, -1, -1\n")

    out.write(frame)
    prev_frame = frame

cap.release()
out.release()
write_results_to_txt(track_eval_format, 's03_with_deepsort.txt')

print("Tracking completed with DeepSORT and optical flow.")