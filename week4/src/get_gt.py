import cv2
import numpy as np

# Configuració
GT_FILE = "/ghome/c5mcv01/mcv-c6-2025-team1/week4/src/tracking/TrackEval/data/trackers/mot_challenge/week3-train/score_091/data/c010.txt"
VIDEO_FILE = "/ghome/c5mcv01/mcv-c6-2025-team1/data/mct_data/train/S03/c010/vdo.avi"
OUTPUT_FILE = "approach1_c010.avi"
FPS = 10  # Per defecte

# Llegir les anotacions
detections = {}
with open(GT_FILE, "r") as f:
    for line in f:
        parts = list(map(lambda x: int(float(x)), line.strip().split(',')))
        frame_id, obj_id, x, y, w, h = parts[:6]
        if frame_id not in detections:
            detections[frame_id] = []
        detections[frame_id].append((obj_id, x, y, w, h))

# Obrir el vídeo
cap = cv2.VideoCapture(VIDEO_FILE)
if not cap.isOpened():
    print("Error obrint el vídeo")
    exit()

# Obtenir paràmetres del vídeo
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter(OUTPUT_FILE, cv2.VideoWriter_fourcc(*'XVID'), FPS, (frame_width, frame_height))

frame_id = 1  # Les anotacions comencen en 1
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    if frame_id in detections:
        for obj_id, x, y, w, h in detections[frame_id]:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 200), 4)
            (text_width, text_height), baseline = cv2.getTextSize(str(obj_id), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 4)
            cv2.rectangle(frame, (x, y - text_height - 5), (x + text_width, y), (0, 0, 0), -1)
            cv2.putText(frame, str(obj_id), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 200), 4)
    
    out.write(frame)
    frame_id += 1

cap.release()
out.release()
cv2.destroyAllWindows()
print("Procés complet. Vídeo guardat com", OUTPUT_FILE)
