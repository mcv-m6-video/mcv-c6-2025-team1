import cv2
import numpy as np
import argparse

from gaussian_modelling.non_adaptive import GaussianModelling


if __name__ == "__main__":
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--video", help="Path to the video file")
    parser.add_argument("-p", "--bg_percentage", help="Percentage of frames to use for background modelling", default=0.25)
    parser.add_argument("-a", "--alpha", help="Alpha parameter for the Gaussian Modelling algorithm", default=0.01)
    parser.add_argument("-o", "--output", help="Path to the output video file", default="output.avi")
    parser.add_argument("-m", "--mask", help="Path to the output mask file", default="mask.avi")
    parser.add_argument("-t", "--area_threshold", help="Minimum area of the bounding box", default=100)
    args = parser.parse_args()
    
    # Get video 
    video_path = args.video
    bg_percentage = float(args.bg_percentage)
    alpha = float(args.alpha)
    output_path = args.output
    mask_path = args.mask
    area_threshold = float(args.area_threshold)
    
    # Read the video
    cap = cv2.VideoCapture(video_path)
    
    # Create video writer with same specs as the original video
    cap_out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), 30, (int(cap.get(3)), int(cap.get(4))))
    mask_out = cv2.VideoWriter(mask_path, cv2.VideoWriter_fourcc(*'XVID'), 30, (int(cap.get(3)), int(cap.get(4))))
    
    # Get number of frames
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_bg_frames = int(bg_percentage * num_frames)
    print(f"Number of frames: {num_frames}")
    print(f"Number of background frames: {n_bg_frames}")

    gaussian_modelling = GaussianModelling(alpha=alpha)
    bg_frames = []
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
            if frame_number == n_bg_frames:
                gaussian_modelling.get_bg_model(np.array(bg_frames))
        else:
            mask = gaussian_modelling.get_mask(frame)
            bounding_box, output_frame = gaussian_modelling.get_bounding_box(mask, frame, area_threshold=area_threshold)
            
            # Save the frame
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            cap_out.write(output_frame)
            mask_out.write(mask)

    cap.release()
    cap_out.release()
    mask_out.release()