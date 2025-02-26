import cv2
import numpy as np
import argparse

import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add the parent directory to sys.path
sys.path.append(parent_dir)

from utils import read_annotations, mean_avg_precision
from gaussian_modelling.base import GaussianModelling


if __name__ == "__main__":
    
    # Read the video
    video_path = "/ghome/c3mcv02/mcv-c6-2025-team1/data/AICity_data/train/S03/c010/vdo.avi"
    output_dir = "/ghome/c3mcv02/mcv-c6-2025-team1/src/visualization"
    cap = cv2.VideoCapture(video_path)
    
    # Get number of frames
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_bg_frames = int(0.25 * num_frames)
    print(f"Number of frames: {num_frames}")
    print(f"Number of background frames: {n_bg_frames}")

    gaussian_modelling = GaussianModelling(alpha=3.5, use_median=False)
    bg_frames = []
    metrics = []
    while True:
        # Read the frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get the frame number
        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        print(f"Processing frame {frame_number}/{num_frames}")
        
        if frame_number <= n_bg_frames:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            bg_frames.append(np.array(gray_frame))

            if frame_number <= 3:
                # Construct frame filename
                frame_filename = os.path.join(output_dir, f"frame_{frame_number:04d}.jpg")  # Saves as frame_0000.jpg, frame_0001.jpg, etc.
                # Save the frame as an image
                cv2.imwrite(frame_filename, gray_frame)

            if frame_number == n_bg_frames:
                gaussian_modelling.get_bg_model(np.array(bg_frames))
                frame_filename = os.path.join(output_dir, f"frame_{frame_number:04d}.jpg")  # Saves as frame_0000.jpg, frame_0001.jpg, etc.
                cv2.imwrite(frame_filename, gray_frame)
                frame_filename = os.path.join(output_dir, f"mean.jpg")  # Saves as frame_0000.jpg, frame_0001.jpg, etc.
                cv2.imwrite(frame_filename, gaussian_modelling.mean)

                # Plot heatmap of pixel variance
                frame_width = 1920
                frame_height = 1080
                plt.figure(figsize=(frame_width / 100, frame_height / 100))
                ax = sns.heatmap(np.sqrt(gaussian_modelling.variance), cmap="viridis", xticklabels=False, yticklabels=False)
                # Add colorbar
                cbar = ax.collections[0].colorbar
                cbar.set_label("Variance")
                # Set title
                plt.title("Pixel Variance Heatmap")
                # Save the heatmap as an image
                output_path = "variance_heatmap.png"
                plt.savefig(output_dir, dpi=300, bbox_inches="tight")
                
        else:
            mask = gaussian_modelling.get_mask(frame, opening_size=13, closing_size=3)  
            if frame_number > 2139:
                frame_filename = os.path.join(output_dir, f"outputframe.jpg")  # Saves as frame_0000.jpg, frame_0001.jpg, etc.
                cv2.imwrite(frame_filename, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) )
           

    cap.release()