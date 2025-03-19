<h2 align="center">WEEK 4: Final presentation </h2>

## Table of Contents

- [Fine-tuning detections](#fine-tuning-detections)
- [Multi-Target Single-Camera Tracking](#multi-target-single-camera-tracking)
- [Multi-Target Multi-Camera Tracking: Re-identification algorithm](#multi-target-multi-camera-tracking-re-identification-algorithm)

## Fine-tuning detections 

### Dataset Download

The dataset can be downloaded from the following link:  
[AIC19 Track1 MTMC Train](https://imatge.upc.edu/web/sites/default/files/projects/teaching/mcv-m6-video/aic19-track1-mtmc-train.zip)  

Once downloaded, it should be placed in the following directory: 

`/mcv-c6-2025-team1/data/mct_data`

### Dataset Splitting
To perform fine-tuning of detections, we divided sequences **S01, S03, and S04** into training, validation, and test sets, ensuring an 80%-20% split between training and validation:

- **Training Set**: 22 cameras from S01 & S04, totaling 16,957 frames.
  - Cameras from S01: `c001, c003, c004, c005`
  - Cameras from S04: `c016, c018, c019, c020, c021, c022, c023, c025, c026, c027, c028, c029, c030,c031, c032, c033, c035, c036, c038, c039`
- **Validation Set**: 6 cameras from S01 & S04, totaling 4,094 frames.
  - Cameras from S01: `c002`
  - Cameras from S04: `c017, c024, c034, c037, c040`
- **Test Set**: 6 cameras from S03, totaling 13,517 frames.
  - Cameras from S03: `c010, c011, c012, c013, c014, c015`

### Annotation format
To use the data with YOLO, we converted the annotations provided by the **AI CITY Challenge** into YOLO format. The conversion process involved:
1. Reading the annotations from the Ground Truth file.
2. Normalizing the bounding box coordinates based on image dimensions.
3. Writing the results into label files in YOLO format.
4. Extracting frames from videos at a specific FPS rate.

To run the annotation conversion script, use the following command:
```bash
cd week4/src/detection/
python gt.py --video path/to/video.mp4 \
                          --gt path/to/ground_truth.txt \
                          --output_images path/to/images \
                          --output_labels path/to/labels \
                          --fps 10
```

#### Example:

To convert the video from camera `c001` in sequence `S01`, use the following command:

```bash
python gt.py --video /mcv-c6-2025-team1/data/mct_data/train/S01/c001/vdo.avi \
                        --gt /mcv-c6-2025-team1/data/mct_data/train/S01/c001/gt/gt.txt \
                        --output_images /mcv-c6-2025-team1/week4/src/detection/data/images/train/c001 \
                        --output_labels /mcv-c6-2025-team1/week4/src/detection/data/labels/train/c001 \
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

#### Example:
```bash
python get_detections.py --base_path /mcv-c6-2025-team1/data/mct_data/train/S03 --output_dir /mcv-c6-2025-team1/week4/src/detection/results_detections/S03
```

#### Output Format
The detection results are stored in `.txt` files, with each file corresponding to a processed video. The output files are named following this pattern:

`detections_<camera_id>.txt`

For example, if processing **camera c001**, the output file will be:

`detections_c001.txt`

Each line in the file represents a detected object in a specific frame and follows this format:

`<frame_id>, -1, <x1>, <y1>, <width>, <height>, <confidence>`


Where:
- **`frame_id`** → Frame number where the detection occurs.
- **`-1`** → Placeholder for an object ID (used for tracking, but remains `-1` in detection-only mode).
- **`x1, y1`** → Coordinates of the top-left corner of the bounding box.
- **`width, height`** → Dimensions of the bounding box.
- **`confidence`** → Detection confidence score (between 0 and 1).

### Evaluating the YOLO Detector  

To assess the quantitative performance of the fine-tuned YOLO model, run the following command:  

```bash
python evaluation.py --m /mcv-c6-2025-team1/week4/src/detection/models/yolo_best.pt
```

Result:

```bash
Average Precision per class:
  person: 0.9167
  bicycle: 0.9144
  car: 0.9104
```

This evaluation provides the Average Precision (AP) for each class in the dataset, measuring the model’s ability to detect objects accurately.

## Multi-Target Single-Camera Tracking
After obtaining the detections for each camera, we applied **Multi-Target Single-Camera Tracking** using the StrongSORT algorithm, which is implemented in the **BoxMot** [repository](https://github.com/mikel-brostrom/boxmot). 

### StrongSORT Algorithm Overview
StrongSORT is a powerful tracking algorithm designed to track multiple objects (i.e., vehicles, pedestrians, etc.) across frames in a video. It builds upon the SORT (Simple Online and Realtime Tracking) algorithm, which uses Kalman Filters and the Hungarian algorithm for object association. StrongSORT, however, introduces improvements, including the use of appearance features for more robust tracking. Here's a breakdown of how the algorithm works:

1. Kalman Filter: The Kalman Filter is used to estimate the state of an object (position and velocity) in each frame. It makes predictions about the object's state even if no detection is made in the current frame, providing smoother tracking.
2. Detection Association: The algorithm associates the current frame's detections with previously tracked objects using the Hungarian algorithm, which solves the assignment problem of matching detections with existing tracks.
3. Appearance Features: StrongSORT enhances the original SORT algorithm by integrating appearance features. These are learned representations of the objects (e.g., appearance embeddings) that help in distinguishing objects of similar size or in cluttered scenes. This is done through a deep learning model, which makes the tracker more resilient to occlusions and misdetections.
4. Re-ID (Re-Identification): This feature is crucial for tracking objects that temporarily leave the camera's field of view and re-enter later. The appearance features allow StrongSORT to identify objects even after they have been lost for a short time.
5. Output: The result of applying StrongSORT is a unique ID for each object across frames, along with its position and trajectory, enabling reliable multi-object tracking over time.

### Using StrongSORT in BoxMot
In the **BoxMot** implementation, the StrongSORT is already integrated into the tracking pipeline. By running the tracking script with the detection results, we obtain continuous tracking of vehicles across the video frames.
To execute the multi-target single-camera tracking, we simply need to provide the detection results from the fine-tuned YOLO model, and StrongSORT will handle the tracking process.

To execute the multi-target tracking with the **StrongSORT** algorithm in **BoxMOT**, the following command is used:
```bash
cd week4/src/tracking/
python3 -m src.tracking.tracking_boxmot \
            -d /path/to/detections.txt \
            -v /path/to/video.avi \
            -ov /path/to/output_video.avi \
            -o /path/to/output_tracking_results.txt \
            --device cuda --tracking_method=strongsort \
            --reid_model 'clip_vehicleid.pt' \
            -c /path/to/strongsort.yaml
```
For the ReID Model, we used the [CLIP-ReID model](https://github.com/Syliz517/CLIP-ReID), an innovative approach that levergaes Vision-Language Pretraining for image re-identification without relying on explicit textual labels. This model was pre-trained on [VehicleID](https://www.pkuml.org/resources/pku-vehicleid.html) dataset,which comprises images captured from multiple surveillance cameras in a small Chinese city, totaling 221,763 images of 26,267 vehicles. 

#### Example:

For a specific example with data from a camera, you can use the following command:

```bash
python3 tracking_boxmot.py \
            -d /mcv-c6-2025-team1/week4/src/detection/results_detections/S03/detections_c010.txt \
            -v /mcv-c6-2025-team1/data/mct_data/train/S03/c010/vdo.avi \
            -ov /mcv-c6-2025-team1/week4/src/tracking/results_tracking/S03/tracking_S03_c010.avi \
            -o /mcv-c6-2025-team1/week4/src/tracking/results_tracking/S03/tracking_S03_c010_results.txt \
            --device cuda --tracking_method=strongsort \
            --reid_model 'clip_vehicleid.pt' \
            -c /mcv-c6-2025-team1/week4/src/tracking/configs/strongsort.yaml
```

You can watch the example tracking video at the following Google Drive link:  
[tracking_S03_c010.avi](https://drive.google.com/file/d/1wNUwlbY453TlgVQi4OgrDGGoWbtI4hUZ/view?usp=sharing)


## Multi-Target Multi-Camera Tracking: Re-identification algorithm

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
cd week4/src/algoritm_features/
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
