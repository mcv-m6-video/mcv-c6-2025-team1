
import os
import subprocess

# Paths base
data_path = "/ghome/c5mcv01/mcv-c6-2025-team1/data/train/S04"
detections_base = "/ghome/c5mcv01/mcv-c6-2025-team1/week3/src/multi_target_tracking/detections_yolo_2_3"
output_video_base = "/ghome/c5mcv01/mcv-c6-2025-team1/week3/src/multi_target_tracking/videos_strongsort"
output_txt_base = "/ghome/c5mcv01/mcv-c6-2025-team1/week3/src/multi_target_tracking/tracking_results_stronsort"
config_path = "/ghome/c5mcv01/mcv-c6-2025-team1/week3/configs/strongsort.yaml"

# Llistar totes les carpetes dins de S03
for folder_name in sorted(os.listdir(data_path)):
    folder_path = os.path.join(data_path, folder_name)
    
    # Comprovar si és un directori
    if os.path.isdir(folder_path):
        
        # Definir paths específics
        video_path = os.path.join(folder_path, "vdo.avi")
        detections_path = os.path.join(detections_base, f"detections_{folder_name}.txt")
        output_video_path = os.path.join(output_video_base, f"{folder_name}.avi")
        output_txt_path = os.path.join(output_txt_base, f"{folder_name}.txt")
        
        # Construir el comandament
        command = [
            "python3", "-m", "src.tracking.tracking_boxmot",
            "-d", detections_path,
            "-v", video_path,
            "-ov", output_video_path,
            "-o", output_txt_path,
            "--device", "cuda",
            "--tracking_method=strongsort",
            "--alpha", "1.0",
            "--pred_method", "mean",
            "-of", "rapidflow",
            "-c", config_path
        ]
        
        # Executar el comandament
        print(f"Executant tracking per {folder_name}...")
        subprocess.run(command)