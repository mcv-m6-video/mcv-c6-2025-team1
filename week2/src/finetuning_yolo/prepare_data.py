from utils import read_annotations
import os


IMG_WIDTH = 1920
IMG_HEIGHT = 1080

if __name__ == '__main__':
    # Read annotations
    annotations = read_annotations('/ghome/c3mcv02/mcv-c6-2025-team1/data/ai_challenge_s03_c010-full_annotation.xml')
    
    # Directory where you want to save YOLO annotation files
    annotations_dir = '/ghome/c3mcv02/mcv-c6-2025-team1/data/train/labels'
    os.makedirs(annotations_dir, exist_ok=True)
    
    for frame_id, bboxes in annotations.items():
        # Create a text file for the current frame (e.g., 690.txt for frame '690')
        file_path = os.path.join(annotations_dir, f"{int(frame_id) + 1}.txt")
        with open(file_path, 'w') as f:
            for bbox in bboxes:
                # Optionally, you could filter by occlusion. For example, if
                # you only want to include non-occluded objects, uncomment:
                # if bbox['occluded'] == 1:
                #     continue

                xtl = bbox['xtl']
                ytl = bbox['ytl']
                xbr = bbox['xbr']
                ybr = bbox['ybr']

                # Calculate box center, width, and height
                x_center = (xtl + xbr) / 2.0
                y_center = (ytl + ybr) / 2.0
                box_width = xbr - xtl
                box_height = ybr - ytl

                # Normalize the coordinates
                x_center_norm = x_center / IMG_WIDTH
                y_center_norm = y_center / IMG_HEIGHT
                width_norm = box_width / IMG_WIDTH
                height_norm = box_height / IMG_HEIGHT

                # Class 0 will be car (the only class in this dataset)
                class_id = 0

                # Create a line in YOLO format: class x_center y_center width height
                line = f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}\n"
                f.write(line)
        print(f"Wrote annotations for frame {frame_id} to {file_path}")