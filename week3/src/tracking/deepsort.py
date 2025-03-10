import cv2
import numpy as np
import argparse
import time

from src.tracking.utils import write_results_to_txt, read_detections_from_txt
from boxmot import DeepOCSORT  # Import DeepOCSORT from boxmot
from src.optical_flow.of import OpticalFlow

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepOCSORT tracking with Optical Flow.")
    parser.add_argument("-d", "--detection_file_path", required=True, type=str, help="Path to the object detections TXT file.")
    parser.add_argument("-v", "--video_path", required=True, type=str, help="Path to the video file.")
    parser.add_argument("-ov", "--output_video_path", required=True, type=str, help="Path to the output video.")
    parser.add_argument("-o", "--output_path", required=True, type=str, help="Path to TXT file where the results will be stored")
    parser.add_argument("-m", "--of_model", default="rpknet", type=str, choices=["pyflow", "diclflow", "memflow", "rapidflow", "rpknet", "dip"], help="Optical flow model to use.")
    args = parser.parse_args()

    # Cargar detecciones desde el archivo
    detections, detections_vect = read_detections_from_txt(args.detection_file_path)

    # Inicializar DeepOCSORT
    tracker = DeepOCSORT()

    # Abrir el video
    cap = cv2.VideoCapture(args.video_path)

    # Obtener información del video
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Preparar el writer para el video de salida
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(args.output_video_path, fourcc, fps, (frame_width, frame_height))

    # Inicializar Optical Flow
    prev_frame = None
    optical_flow = OpticalFlow(args.of_model)

    # Procesar cada frame
    track_eval_format = []
    while cap.isOpened():
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        print(f"Processing frame {frame_number}")

        # Obtener detecciones del frame actual
        actual_bb = [detection[1:5] for detection in detections_vect if detection[0] == frame_number]
        actual_scores = [detection[5] for detection in detections_vect if detection[0] == frame_number]

        # Convertir detecciones al formato DeepOCSORT [x1, y1, x2, y2]
        actual_bb = np.array(actual_bb)
        if len(actual_bb) > 0:
            actual_bb[:, 2] += actual_bb[:, 0]  # x2 = x1 + w
            actual_bb[:, 3] += actual_bb[:, 1]  # y2 = y1 + h

            # Aplicar Optical Flow para ajustar bounding boxes
            if prev_frame is not None:
                flow = optical_flow.compute_flow(prev_frame, frame_gray)
                for i, bbox in enumerate(actual_bb):
                    x1, y1, _, _ = bbox
                    flow_x = flow[int(y1), int(x1)][0]  # Componente x del flujo óptico
                    flow_y = flow[int(y1), int(x1)][1]  # Componente y del flujo óptico
                    actual_bb[i, 0] += flow_x
                    actual_bb[i, 1] += flow_y

            # Preparar detecciones en el formato (x1, y1, x2, y2, confidence, class)
            detections_for_tracker = np.hstack([actual_bb, np.array(actual_scores).reshape(-1, 1), np.zeros((len(actual_bb), 1))])  # Assume class = 0 for all

            # Actualizar el tracker con las detecciones
            tracker_outputs = tracker.update(detections_for_tracker, frame_bgr)

        else:
            tracker_outputs = tracker.update(np.empty((0, 6)), frame_bgr)

        # Dibujar bounding boxes y track IDs
        for obj in tracker_outputs:
            x1, y1, x2, y2, track_id, _, _ = map(int, obj)  # Unpack the output
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame_bgr, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Guardar tracking info en formato de evaluación
            track_eval_format.append(f"{frame_number}, {track_id}, {x1}, {y1}, {x2-x1}, {y2-y1}, -1, -1, -1, -1\n")

        # Guardar frame con anotaciones
        out.write(frame_bgr)
        prev_frame = frame_gray

    # Liberar recursos
    cap.release()
    out.release()

    # Guardar los resultados en TXT
    write_results_to_txt(track_eval_format, args.output_path)
    print("Annotated video with DeepOCSORT and Optical Flow saved.")
