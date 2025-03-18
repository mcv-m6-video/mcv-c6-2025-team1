<h2 align="center">WEEK 4: Final presentation </h2>

## Table of Contents

- [Fine-tuning detections](#fine-tuning-detections)


## Fine-tuning detections 

### Dataset Splitting
To perform fine-tuning of detections, we divided sequences S01, S03, and S04 into training, validation, and test sets, ensuring an 80%-20% split between training and validation:
- **Training Set**: 22 cameras from S01 & S04, totaling 16,957 frames.
- **Validation Set**: 6 cameras from S01 & S04, totaling 4,094 frames.
- **Test Set**: 6 cameras from S03, totaling 13,517 frames.

### Detector Selection
We chose YOLO11x as our detector and we trained the entire network using the training set.

### Annotation format
To use the data with YOLO, we converted the annotations provided by the AI CITY Challenge into YOLO format. This process involves:
1. Reading the annotations from the Ground Truth file.
2. Normalizing the bounding box coordinates based on image dimensions.
3. Writing the results into label files in YOLO format.
4. Extracting frames from videos at a specific FPS rate.

To run the script for converting annotations, run:
```bash
python gt.py --video path/to/video.mp4 \
                          --gt path/to/ground_truth.txt \
                          --output_images path/to/images \
                          --output_labels path/to/labels \
                          --fps 10
```
