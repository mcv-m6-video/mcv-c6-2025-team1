import cv2
import numpy as np
import optuna
from utils import read_annotations, mean_avg_precision
from gaussian_modelling.adaptive import AdaptiveGaussianModelling

def optimize_model(trial):
    # Define the search space for each hyperparameter
    alpha = trial.suggest_float('alpha', 3, 4)  # Alpha for Gaussian Modelling between 1 and 15
    rho = trial.suggest_float('rho', 0.001, 0.1)  # Rho Gaussian Modelling between 0 and 1
  
    # Print the parameters being tried
    print(f"Trying parameters: alpha={alpha}, rho={rho}")

   
    
    # Read the video
    cap = cv2.VideoCapture("/ghome/c3mcv02/mcv-c6-2025-team1/data/AICity_data/train/S03/c010/vdo.avi")  # Put the correct path to the video
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_bg_frames = int(0.25 * num_frames)  # Percentage of frames for background modeling
    
    # Read the ground truth annotations
    gt_boxes = read_annotations("/ghome/c3mcv02/mcv-c6-2025-team1/data/ai_challenge_s03_c010-full_annotation.xml")  # Put the correct path to the annotations
    
    # Initialize the background model
    gaussian_modelling = AdaptiveGaussianModelling(alpha=alpha, rho=rho, use_median=True)
    bg_frames = []
    aps = []
    
    # Iterate through the frames of the video
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        if frame_number <= n_bg_frames:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            bg_frames.append(np.array(gray_frame))
            if frame_number == n_bg_frames:
                gaussian_modelling.get_bg_model(np.array(bg_frames))
        else:
            mask = gaussian_modelling.get_mask(frame,opening_size=3, closing_size=13)
            bounding_box, output_frame = gaussian_modelling.get_bounding_box(
                mask,
                frame,
                area_threshold=959,
                aspect_ratio_threshold=1.2
                
            )
            
            gaussian_modelling.update_model(frame, mask)

            # Convert the bounding boxes to the proper format
            gt_box = gt_boxes.get(str(frame_number), [])
            gt_box = [list(map(int, [box["xtl"], box["ytl"], box["xbr"], box["ybr"]])) for box in gt_box]
            pred_box = [[int(box[0][0]), int(box[0][1]), int(box[1][0]), int(box[1][1])] for box in bounding_box]
            
            # Calculate the mean average precision
            ap = mean_avg_precision(gt_box, pred_box)
            aps.append(ap)
    
    # Calculate the mean mAP for this combination of parameters
    mean_ap = np.mean(aps)
    cap.release()

    return mean_ap


def main():
    # Definir el cercador d'Optuna
    study = optuna.create_study(direction="maximize")
    study.optimize(optimize_model, n_trials=50)  # Nombre de proves que realitzarà Optuna

    # Mostrar el millor conjunt de paràmetres
    print("Best trial:")
    print(f"  Value: {study.best_value}")
    print(f"  Params: {study.best_params}")

if __name__ == "__main__":
    main()
