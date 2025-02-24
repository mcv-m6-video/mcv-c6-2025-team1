import cv2
import numpy as np
import argparse

from utils import read_annotations, mean_avg_precision
from gaussian_modelling.adaptive import AdaptiveGaussianModelling


if __name__ == "__main__":
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--video", help="Path to the video file")
    parser.add_argument("-p", "--bg_percentage", help="Percentage of frames to use for background modelling", default=0.25)
    parser.add_argument("-a", "--alpha", help="Alpha parameter for the Gaussian Modelling algorithm", default=0.01)
    parser.add_argument("-rho", "--rho", help="Rho parameter for the Gaussian Modelling algorithm", default=0.01)
    parser.add_argument("-o", "--output", help="Path to the output video file", default="output_adaptive.avi")
    parser.add_argument("-m", "--mask", help="Path to the output mask file", default="mask_adaptive.avi")
    parser.add_argument("-t", "--area_threshold", help="Minimum area of the bounding box", default=100)
    parser.add_argument("-r", "--aspect_ratio", help="Aspect Ratio", default=1.0)
    parser.add_argument("-g", "--annotations", help="Path to the ground truth annotations", default=None)
    parser.add_argument("--use_median", help="Use median instead of mean for the background model", action="store_true", default=False)
    parser.add_argument("--opening_size", help="Size of the kernel for the opening operation", default=5)
    parser.add_argument("--closing_size", help="Size of the kernel for the closing operation", default=5)
    args = parser.parse_args()
    
    # Get video 
    video_path = args.video
    bg_percentage = float(args.bg_percentage)
    alpha = float(args.alpha)
    rho = float(args.rho)
    output_path = args.output
    mask_path = args.mask
    area_threshold = float(args.area_threshold)
    aspect_ratio_threshold = float(args.aspect_ratio)
    annotations_path = args.annotations
    opening_size = int(args.opening_size)
    closing_size = int(args.closing_size)
    use_median = args.use_median
    
    # Read the video
    cap = cv2.VideoCapture(video_path)
    
    # Create video writer with same specs as the original video
    cap_out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), int(cap.get(cv2.CAP_PROP_FPS)), (int(cap.get(3)), int(cap.get(4))))
    mask_out = cv2.VideoWriter(mask_path, cv2.VideoWriter_fourcc(*'XVID'), int(cap.get(cv2.CAP_PROP_FPS)), (int(cap.get(3)), int(cap.get(4))))
    
    # Read ground truth annotations
    if annotations_path:
        gt_boxes = read_annotations(annotations_path)
    
    # Get number of frames
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_bg_frames = int(bg_percentage * num_frames)
    print(f"Number of frames: {num_frames}")
    print(f"Number of background frames: {n_bg_frames}")

    gaussian_modelling = AdaptiveGaussianModelling(alpha=alpha, rho=rho, use_median=use_median)
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
            if frame_number == n_bg_frames:
                gaussian_modelling.get_bg_model(np.array(bg_frames))
                print(f"BG model done")
        else:
            mask = gaussian_modelling.get_mask(frame, opening_size=opening_size, closing_size=closing_size)
            bounding_box, output_frame = gaussian_modelling.get_bounding_box(
                mask, 
                frame, 
                area_threshold=area_threshold,
                aspect_ratio_threshold=aspect_ratio_threshold
            )
            # If a pixel is bg --> update the mean and variance (of the Gaussian Model) for that pixel 
            print(f"FG")
            gaussian_modelling.update_model(frame, mask)

            # Convert 2DBB to the same format
            gt_box = gt_boxes.get(str(frame_number), [])
            gt_box = [list(map(int, [box["xtl"], box["ytl"], box["xbr"], box["ybr"]])) for box in gt_box]
            pred_box = [[int(box[0][0]), int(box[0][1]), int(box[1][0]), int(box[1][1])] for box in bounding_box]
            
            # Calculate the mean average precision, precision and recall
            metrics.append(mean_avg_precision(gt_box, pred_box))
            
            # Print GT boxes if they exist
            if gt_boxes:
                try:
                    for box in gt_boxes[str(frame_number)]:
                        xtl, ytl, xbr, ybr = int(box["xtl"]), int(box["ytl"]), int(box["xbr"]), int(box["ybr"])
                        cv2.rectangle(output_frame, (xtl, ytl), (xbr, ybr), (0, 255, 0), 2)
                except KeyError:
                    pass
            
            # Save the frame
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            cap_out.write(output_frame)
            mask_out.write(mask)

    print(f"Mean Average Precision: {np.mean(metrics)}")
    cap.release()
    cap_out.release()
    mask_out.release()