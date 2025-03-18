<h2 align="center">WEEK 4: Final presentation </h2>

## Table of Contents

- [Fine-tuning detections](#fine-tuning-detections)


## Fine-tuning detections 

### Dataset Splitting
To perform fine-tuning of detections, we divided sequences S01, S03, and S04 into training, validation, and test sets, ensuring an 80%-20% split between training and validation:
- **Training Set**: 22 cameras from S01 & S04, totaling 16,957 frames.
- **Validation Set**: 6 cameras from S01 & S04, totaling 4,094 frames.
- **Test Set**: 6 cameras from S03, totaling 13,517 frames.

### Annotation format
To use the data with YOLO, we converted the annotations provided by the **AI CITY Challenge** into YOLO format. The conversion process involved:
1. Reading the annotations from the Ground Truth file.
2. Normalizing the bounding box coordinates based on image dimensions.
3. Writing the results into label files in YOLO format.
4. Extracting frames from videos at a specific FPS rate.

To run the annotation conversion script, use the following command:
```bash
python gt.py --video path/to/video.mp4 \
                          --gt path/to/ground_truth.txt \
                          --output_images path/to/images \
                          --output_labels path/to/labels \
                          --fps 10
```

### Training the Detector
We fine-tuned the **YOLO11x** model using the Ultralytics implementation. The model was loaded from the pre-trained weights file yolo11x.pt. The training process involved 50 epochs with an early stopping mechanism (patience=5) to avoid overfitting. 

To execute the fine-tuning process, use the following command:
```bash
python finetune.py
```
Once the model is fine-tuned, we used it to perform the detections for all the three sequences. To do so, we implemented a detection script that applies the trained model to video frames and outputs the detected objects (such as cars and trucks) along with their bounding boxes and confidence scores.

The following command is used to run the detection script:
```bash
python get_detections.py --base_path /path/to/videos --output_dir /path/to/output
```
## Re-identification algorithm

### Overview

The re-identification algorithm in `algorithm.py` is designed to match and track individuals across multiple camera views. It uses appearance features extracted from a CNN model to establish identity correspondences between camera views while accounting for timestamp offsets.

### Algorithm Type

This is a feature-based multi-camera tracking (MCT) algorithm that utilizes:

1. Deep learning feature extraction
2. Cosine similarity matching
3. Temporal constraints
4. Camera synchronization

### Process Description

The re-identification algorithm works through a sequential pipeline that transforms individual per-camera detections into globally consistent identities across multiple viewpoints:

The process begins with **Feature Extraction**, where the system processes each detection across all cameras by loading its corresponding video frame, extracting the precise region defined by the bounding box, and passing this region through a ResNet-50 feature extractor. This produces a distinctive feature vector that characterizes the appearance of each detection, which is then stored alongside the detection metadata.

Rather than comparing individual detections directly, the algorithm implements **Track Representation** by grouping detections that belong to the same track ID within each camera. For each track, it computes an average feature vector from all its constituent detections, creating a more robust representation that mitigates issues with individual poor-quality frames. Simultaneously, it calculates the average timestamp for each track to facilitate temporal alignment.

The core of the algorithm is the **Cross-Camera Association** process, which forms connections between tracks from different cameras. This works by computing pairwise cosine similarity between all track features across cameras, then filtering potential matches using both a similarity threshold (defaulting to 0.9) and a temporal constraint that limits the maximum time difference between tracks (defaulting to 10 seconds). The algorithm explicitly prevents matching tracks from the same camera, and when multiple candidate matches exist from a single camera, it selects the one with the smallest temporal difference to ensure temporal coherence.

Once track associations are established, the **Global ID Assignment** phase assigns a unique global identifier to each cluster of associated tracks. All detections belonging to tracks within a cluster inherit this global ID, creating a consistent identity across multiple cameras. Tracks that couldn't be confidently matched to any others receive a global ID of -1, effectively filtering them out of the final results.

Finally, during **Output Generation**, the algorithm produces detection files containing only the positively identified matches (those with global IDs > 0). Each line in these output files follows the format: `frame_num,global_id,x,y,width,height,confidence,-1,-1,-1`, providing all necessary information for downstream multi-camera tracking applications.

### CLI Usage

```bash
python algorithm.py --detections_folder PATH_TO_DETECTIONS \
                    --videos_folder PATH_TO_VIDEOS \
                    --output_folder PATH_TO_OUTPUT \
                    --camera_offsets PATH_TO_OFFSETS \
                    [--similarity_threshold THRESHOLD] \
                    [--timestamp_threshold SECONDS]
```

#### Required Parameters

- `--detections_folder`: Directory containing detection files (one per camera, named by camera ID)
- `--videos_folder`: Directory containing video files (one per camera, named by camera ID)
- `--output_folder`: Directory where output files with global IDs will be saved
- `--camera_offsets`: Path to a text file containing camera timestamp offsets in format:
  ```
  camera_id offset_value
  ```

#### Optional Parameters

- `--similarity_threshold`: Minimum cosine similarity value to consider a match (default: 0.90)
- `--timestamp_threshold`: Maximum allowed time difference in seconds between tracks (default: 10)

### Performance Considerations

The algorithm's performance depends on several factors:

- Quality of extracted features
- Accuracy of camera synchronization
- Distinctiveness of appearance across views
- Visibility and quality of detections

Modifying the similarity and timestamp thresholds can balance between precision (avoiding false matches) and recall (finding more true matches) based on specific deployment conditions.
