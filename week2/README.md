<h2 align="center">WEEK 2: Tasks</h2>

## Table of Contents

- [Project Structure W2](#project-structure-w2)
- [Task 1: Object detection](#task-1-object-detection)
    - [Task 1.1: Off-the-shelf](#task-11-off-the-shelf)
    - [Task 1.2: Fine-tuning to our data](#task-12-fine-tuning-to-our-data)
    - [Task 1.3: K-Fold Cross Validation](#task-13-k-fold-cross-validation)
- [Task 2: Object tracking](#task-2-object-tracking)
    - [Task 2.1: Tracking by overlap](#task-21-tracking-by-overlap)
    - [Task 2.2: Tracking with Kalman Filter (KF)](#task-22-tracking-with-kalman-filter-kf)
    - [Task 2.3: IDF1, HOTA scores](#task-23-idf1-hota-scores)


### Project Structure W2

    week2/
    ├── checkpoints/
    ├── output_videos/
    ├── src/
    │   ├── models/
    │   │   ├── faster_rcnn.py
    │   │   ├── ssd.py
    │   │   └── ...
    │   ├── detection.py
    │   ├── utils.py
    │   └── ...

## Task 1: Object detection
    
### Task 1.1: Off-the-shelf
This task focuses on implementing and evaluating various pre-trained object detection models (including YOLOv11, Faster R-CNN,DeTR and SSD) for vehicle detection in traffic surveillance videos. Below are the detailed commands and parameters used for each experiment.

For further detail in the detection script please refer to our help command:

```bash
python3 detection.py --help
```

#### YOLOv11 and YOLOv12
Our implementation leverages the YOLOv11n, YOLOv11x, and YOLOv12n models, pre-trained using the Ultralytics framework, originally trained on the COCO dataset. These models are optimized for general object detection across diverse, real-world scenes. We apply these models directly for inference, specifically targeting car and truck classes for vehicle detection.

The YOLOv11n and YOLOv12n models are lightweight versions designed for real-time applications with high computational efficiency, while YOLOv11x is larger and more precise model at the cost of higher computational demand.

You can execute the detection pipeline with:

```bash
python3 detection.py -t yolo \
    -m /path/to/model/yolo11n.pt
    -v /path/to/video/vdo.avi \
    -a /path/to/annotations.xml \
    -o /path/to/output/YOLOv11n.avi
```
The detection script processes each video frame independently, generating bounding box predictions with confidence scores. The results are displayed in real-time and saved to an output video file. Performance metrics, including mAP@0.50, mAP@0.75, are computed and displayer during execution. 


#### Faster R-CNN
Our implementation utilizes PyTorch's pre-trained Faster R-CNN model with a ResNet-50 backbone, originally trained on the COCO dataset. The model architecture consists of a Region Proposal Network (RPN) coupled with a detection network, optimized for general object detection across diverse, real-world scenes. We apply these models directly for inference, specifically for vehicle detection in traffic surveillance scenarios.

For our experiments, we process each frame at original resolution and set a confidence threshold (default of 0.9) for detection filtering. The model focuses exclusively on the car and truck classes (COCO class ID: 3 and 8).

You can execute the detection pipeline with:

```bash
python3 detection.py -t faster-rcnn \
    -b 0.9 \
    -v /path/to/video/vdo.avi \
    -a /path/to/annotations.xml \
    -o /path/to/output/faster_rcnn.avi
```

The detection script processes each video frame independently, generating bounding box predictions with confidence scores. The results are visualized in real-time and saved to an output video file. Performance metrics including mAP@0.50, mAP@0.75, precision and recall are computed and displayed during execution.

#### SSD

Our implementations utilizes two versions of PyTorch's pre-trained SSD (Single Shot Detector) model —one with a VGG-16 backbone and another with a ResNet-50 backbone. Both models were originally trained on the COCO dataset. In the SSD architecture, the backbone (either VGG-16 or ResNet-50) extracts multi-scale convolutional features from the input image. These feature maps are then fed into additional convolutional layers that directly predict object class probabilities and bounding box offsets in a single forward pass, eliminating the need for a separate region proposal stage. These models are optimized for general object detection across diverse, real-world scenes. We apply them directly for inference, specifically targeting car and truck classes for vehicle detection.

For our experiments, we process each frame at original resolution and set a confidence threshold (default of 0.9) for detection filtering. The model focuses exclusively on the car and truck classes (COCO class ID: 3 and 8).

You can execute the detection pipeline with:

(SSD with VGG-16 backbone)
```bash
python3 detection.py -t ssd-vgg16 \
    -b 0.9 \
    -v /path/to/video/vdo.avi \
    -a /path/to/annotations.xml \
    -o /path/to/output/ssd_vgg16.avi
```

(SSD with ResNet-50 backbone)
```bash
python3 detection.py -t ssd-resnet50 \
    -b 0.9 \
    -v /path/to/video/vdo.avi \
    -a /path/to/annotations.xml \
    -o /path/to/output/ssd_resnet50.avi
```

The detection script processes each video frame independently, generating bounding box predictions with confidence scores. The results are visualized in real-time and saved to an output video file. Performance metrics including mAP@0.50, mAP@0.75, precision and recall are computed and displayed during execution.

#### DETR

Our implementation utilizes the pre-trained DEtection TRansformer (DETR) model with a ResNet-50 backbone, originally trained on the COCO dataset. DETR is an end-to-end object detection model based on transformers, eliminating the need for traditional region proposal networks.  

For our experiments, we process each frame at its original resolution and set a confidence threshold (default of 0.9) to filter detections. The model is optimized for general object detection across diverse, real-world scenes, and we apply it directly for inference, specifically targeting car and truck classes for vehicle detection (COCO class IDs: 3 and 8).  

You can execute the detection pipeline with:  

```bash
python3 detection.py -t detr \
    -b 0.9 \
    -v /path/to/video/vdo.avi \
    -a /path/to/annotations.xml \
    -o /path/to/output/detr.avi
```

The detection script processes each video frame independently, generating bounding box predictions with confidence scores. The results are visualized in real-time and saved to an output video file. Performance metrics including mAP@0.50, mAP@0.75, precision and recall are computed and displayed during execution.


#### Quantitative results

This section presents the quantitative results of various object detection models evaluated using **mAP@0.5** and **mAP@0.75**. The **mAP (Mean Average Precision)** values indicate the model's performance in terms of average precision at different IoU thresholds, where **mAP@0.5** corresponds to an IoU of 0.5 and **mAP@0.75** to an IoU of 0.75. 

| Model                                  | mAP@0.5 | mAP@0.75 |
|----------------------------------------|--------|---------|
| **YOLOv11x**                           | 0.52   | 0.47    |
| **YOLOv11n**                           | 0.49   | 0.44    |
| **YOLOv12n**                           | 0.47   | 0.44    |
| **Faster R-CNN (conf. 0.5)**           | 0.57   | 0.44    |
| **Faster R-CNN (conf. 0.7)**           | 0.59   | 0.44    |
| **Faster R-CNN (conf. 0.9)**           | 0.45   | 0.39    |
| **SSD (Backbone: ResNet50, conf. 0.1)** | 0.49   | 0.32    |
| **SSD (Backbone: ResNet50, conf. 0.25)**| 0.39   | 0.31    |
| **SSD (Backbone: ResNet50, conf. 0.5)** | 0.32   | 0.30    |
| **SSD (Backbone: VGG16, conf. 0.25)**   | 0.39   | 0.31    |
| **SSD (Backbone: VGG16, conf. 0.5)**    | 0.30   | 0.29    |
| **SSD (Backbone: VGG16, conf. 0.9)**    | 0.16   | 0.16    |
| **DETR (conf. 0.9)**                    | 0.68 | 0.48  |
| **DETR (conf. 0.7)**                    | 0.75 | 0.49  |
| **DETR (conf. 0.5)**                    | **0.78** | **0.49**  |

Overall, **DeTR** proves to be the most effective model in this comparison, while YOLO remains a competitive choice for applications requiring a good balance between speed and accuracy. 


#### Qualitative results

In all cases depicted below, green boxes always refer to GT boxes and red ones are predicted boxes.

> [!WARNING]
> Some Ground Truth bounding boxes are not correctly placed due to some human errors in the annotations. Take that into account when comparing predicted and GT boxes.

- **YOLOv11:**

The image shown represents one of the first frames in the sequence. Some parked cars in the background are not correctly detected due to occlusions, and several bounding boxes don't align precisely with the Ground Truth annotations, despite the cars themselves being successfully detected.

![image](https://github.com/user-attachments/assets/516e27a8-dbe5-4c5e-84cf-300f9fbded5f)

- **Faster R-CNN:**

The image shown represents one of the first frames in the sequence. Some parked cars in the background are not correctly detected due to occlusions, and several bounding boxes don't align precisely with the Ground Truth annotations, despite the cars themselves being successfully detected.

![image](https://github.com/user-attachments/assets/323513b9-a217-4e8f-86eb-0de5dc544eca)


- **DETR:**

To qualitatively evaluate the **DEtection TRansformer (DETR)** model, we conducted an experiment using the `detr.py` script. This script runs a forward pass of the DETR model on the first frame of a given video and visualizes the detections.  

```bash
python3 detr.py
```

The script processes the input frame by:  
- Loading a pre-trained DETR model with a ResNet-50 backbone.  
- Preprocessing the image and running inference to obtain bounding box predictions.  
- Filtering detections to retain only **cars** and **trucks** based on COCO class IDs (3 and 8).  
- Drawing the predicted bounding boxes on the image.  

Below is an example output, where the model successfully detects vehicles in the first frame of the input video:  

![first_frame_detr](https://github.com/user-attachments/assets/46b3662a-695e-4757-8ce8-5a69f6c56401)

- **SSD:**

Unlike the previous models, **SSD** detects fewer objects in this scenario. In the image shown, it only identifies **two** cars, whereas models like **DETR** manage to detect up to **six** vehicles in the same frame.

This result suggests that SSD might be less sensitive to certain objects in the scene, possibly due to its multi-scale detection approach with predefined anchor boxes, which can affect its performance in scenarios with small or partially occluded objects.

![ssd_fist_frame](https://github.com/user-attachments/assets/adfdda12-24c6-4813-9e0f-8f9581808090)


### Task 1.2: Fine-tuning to our data

#### Data preparation
The `prepare_data` script converts your custom dataset into YOLO's required format for training. The following command will process the data and save the annotations in the specified path:

```bash
python3 -m src.finetuning_yolo.prepare_data -a <ANNOTATIONS_PATH> -y <YOLO_GT_PATH>
```

This script converts annotation data from an XML format to YOLO format for object detection training. It takes an input XML annotations file and outputs YOLO-compatible text files in a specified directory. For each frame in the dataset, the script creates a corresponding text file containing normalized bounding box coordinates (center x, center y, width, height) for each detected object. 

The script assumes all objects belong to a single class (cars) and handles the conversion of coordinate systems from top-left/bottom-right format to the center-based format required by YOLO. Image dimensions are set to 1920x1080 pixels for normalization purposes.

#### Training process
This script implements a comprehensive YOLO object detection training pipeline with three configurable dataset splitting strategies: 
- Strategy A (simple 25/75 train/validation split)
- Strategy B (standard K-fold cross-validation with 4 folds)
- Strategy C (randomized K-fold cross-validation).

The pipeline handles the entire training workflow, including dataset organization, YAML configuration generation, model training using YOLOv11x architecture, and result evaluation. Training runs for 50 epochs with early stopping (patience=20) and batch size of 8, utilizing CUDA acceleration. 

For K-fold strategies, the script automatically trains models across all folds and provides aggregated performance metrics. The implementation is designed to be run with a command-line argument to specify the desired splitting strategy (--strategy A|B|C).

```bash
python3 -m src.finetune_yolo -s <A|B|C> -b <BASE_DIR> -f <SOURCE_FRAMES_PATH> -l <SOURCE_LABELS_PATH> -d <OUTPUT_DATASET_PATH>
```

In here, `SOURCE_FRAMES_PATH` refers to a path where all frames have been extracted from the video, it can be done by using `ffmpeg` with the following command:

```bash
ffmpeg -i <VIDEO_INPUT> -frame_pts 1 <SOURCE_FRAMES_PATH>/frame_%d.jpg
```

#### Results with Strategy A
The following table shows the results for the training strategy A:

| Strategy | Precision | Recall  | mAP50   | mAP50-95 |
|------------------|---------|---------|----------|----------------------------------|
| Fully unfrozen    | 0.98477   | 0.93714 | 0.97813 | 0.91076  |
| Backbone frozen    | 0.98257   | 0.95637 | 0.98614 | 0.91917  |


### Task 1.3: K-Fold Cross Validation

#### K-Fold results evaluation
In order to evaluate K-Fold results, you may use the following script:

```bash
python3 -m src.finetuning_yolo.evaluate_kfold -p <PATH_TO_YOUR_KFOLD_RESULTS> [--is_random]
```

The flag `--is_random` is used to evaluate random fold (if not used, then it will evaluate fixed fold cases). The script will output the K-Fold mean and standard deviation of all metrics as in this table:

Fine tuning strategy | Data Strategy | Precision | Recall | mAP@50 |
|--------------------|----------|-----------|--------|--------|
| Fully unfrozen | B (fixed) | 0.9844 ± 0.0012 | 0.9512 ± 0.0083 | 0.9791 ± 0.0021 |
| Backbone frozen | B (fixed) | 0.9826 ± 0.0014 | 0.9632 ± 0.0050 | 0.9867 ± 0.0004 |
| Fully unfrozen | C (random) | 0.9929 ± 0.0009 | 0.9768 ± 0.0022 | 0.9889 ± 0.0009 |
| Backbone frozen | C (random) | 0.9954 ± 0.0007 |0.9791 ± 0.0018 | 0.9915 ± 0.0013 |

## Task 2: Object tracking
In this task, we focus on object tracking, specially using the **tracking-by-detection** approach. This method relies on object detections obtained through inference from the best-performing model in **Task 1.3 (Strategy C)**. The primary goal is to consistenly track objects across frames while ensuring that each object retains a unique ID throughout the sequence. 

To extract detections, run the following command:

```bash
python tracking/get_detections.py
```

### Task 2.1: Tracking by overlap
The tracking-by-overlap algorithm assigns unique track IDs to objects across frames based on the **Intersection over Union (IoU)** metric. The goal is to track objects consistently over time while ensuring each each object has a unique ID.

1. **Initialization:** The algorithm starts by initializing a counter for **track_id**, which is used to assign unique IDs to objects in the video sequence. The list **active_tracks** holds the active tracked objects, and **track_eval_format** is where the results are stored in the required evaluation format. Finally, **frame_groups** is a dictionary used to group detections by their respective frame IDs.
2. **Grouping detections:** The first step is to group all detections by their corresponding frame IDs. This allows the algorithm to process detections sequentially by frame.
3. **First frame (Initialization of Tracks):** For the first frame, there are no prior tracks to match to, so every detection in this frame is assigned a unique **track_id**. Each detection is added to the **active_tracks** list with its corresponding **track_id** and bounding box.
4. **Subsequent frames (Tracking with IoU):** In subsequent frames, the algorithm matches new detections with existing tracks based on the highest IoU between the bounding boxes. For each detection in the current frame, the algorithm computes the IoU with all active tracks from the previous frames. If the IoU between a detection and an active track is above a threshold (0.5 in this case), the detection is considered a match to that track. The track ID of the matched detection is then updated, and the detection is assigned to that track. If there are multiple matches, the algorithm selects the one with the highest IoU.
5. **New detections:** If a detection does not match any active track, a new unique **track_id** is assigned to that detection.

The results of the tracking process are stored in the **track_eval_format** list. Each entry consists of the frame ID, track ID, bounding box coordinates, confidence score, etc. in the format required for evaluation.

To extract tracking results, run the following command:

```bash
python tracking/overlap.py
```
### Task 2.2: Tracking with Kalman Filter (KF)  

**Tracking with KF** assigns unique track IDs to objects across frames using a **Kalman Filter (KF)** to predict motion and associates detections based on spatial overlap.  

#### Algorithm  
We use the `Sort(object)` class from [`sort.py`](https://github.com/abewley/sort/tree/master). The state of each tracked object is represented as a **7-element vector**:  
`[xc, yc, s, r, vx, vy, vs]`, where `[xc, yc]` is the 2D bounding box (2DBB) center, `s` is the scale (area), `r` is the aspect ratio, and `[vx, vy, vs]` are velocities (aspect ratio remains constant).  

#### Kalman Filter Steps  

The KF approach can be summarized into 2 main steps. For each object:
1. **Prediction:** Estimates object position in the current frame of all active trackers using a linear constant velocity motion model.  
2. **Update:** Computes IoU between active trackers' predicted poses and current detections. Based on IoU, detections are associated/matched to active trackers, so matched trackers' states are updated. New trackers are created for all unmatched detections, and unmatched trackers are removed (if they have not been updated for max. number of consecutive frames).  

#### Initialization Parameters  

- **max_age:** Max frames a tracker stays active without updates.  
- **min_hits:** Min frames an object must be detected to be considered valid.  
- **iou_threshold:** IoU threshold for matching detections to trackers.

The tracking results are stored in the required evaluation format, including **frame ID, track ID, bounding box coordinates, confidence score, etc.**  

To extract tracking results, run the following command:  

```bash
python tracking/tracking_kf.py
```
The initialization parameters can be set to different values in line 50 (`mot_tracker = Sort(max_age = 21, min_hits=3, iou_threshold=0.1)`) of `tracking_kf.py`. 

### Task 2.3: IDF1, HOTA scores
To evaluate the performance of the tracking algorithm, we use the **TrackEval** framework. TrackEval is a tool designed to compute various tracking performance metrics, including **IDF1** and **HOTA** scores, which are commonly used in multi-object tracking (MOT) tasks.

#### **IDF1 (Identification F1 Score):**
IDF1 measures the ability of the tracking algorithm to consistently assign the same track ID to the same object across frames. It takes into account both **precision** and **recall** by evaluating how well the tracking system can identify and maintain the same ID across frames. A higher IDF1 score indicates better tracking consistency.

#### **HOTA (Higher Order Tracking Accuracy):**
HOTA is a comprehensive metric that evaluates tracking performance by combining both **tracking quality** (how well objects are tracked) and **association quality** (how well object detections are associated with true objects). It is considered more robust than other metrics like **MOTA** because it accounts for both false positives and false negatives, as well as ID switches and missed detections.

Before running the evaluation, we organized the data as follows:

    TrackEval/
        ├── data/
        │   ├── gt/
        │   │   ├── mot_challenge/
        │   │   │   ├── seqmaps/
        │   │   │   │   └── week2-train.txt
        │   │   │   ├── week2-train/
        │   │   │   │   ├── s03/
        │   │   │   │   │   ├── seq.info.ini
        │   │   │   │   │   ├── gt/
        │   │   │   │   │   │   └── gt.txt
        │   ├── trackers/
        │   │   ├── mot_challenge/
        │   │   │   ├── week2-train/
        │   │   │   │   ├── kalman/
        │   │   │   │   │   └── data/
        │   │   │   │   │       └── s03.txt
        │   │   │   │   ├── overlap/
        │   │   │   │   │   └── data/
        │   │   │   │   │       └── s03.txt
        │   └── ...

The above structure is essential for **TrackEval** to correctly locate the ground truth and tracker output files for comparison.

To evaluate the tracker's performance, the following comand was executed:

```bash
python tracking/TrackEval/scripts/run_mot_challenge.py \
  --GT_FOLDER /ghome/c3mcv02/mcv-c6-2025-team1/week2/src/tracking/TrackEval/data/gt/mot_challenge \
  --TRACKERS_FOLDER /ghome/c3mcv02/mcv-c6-2025-team1/week2/src/tracking/TrackEval/data/trackers/mot_challenge \
  --BENCHMARK week2 \
  --SEQ_INFO s03 \
  --DO_PREPROC=False
```
**Parameters:**
- **`--GT_FOLDER`**: Path to the ground truth annotations (actual object positions and IDs).
- **`--TRACKERS_FOLDER`**: Path to the tracker’s output (predicted object positions and IDs).
- **`--BENCHMARK`**: The benchmark (e.g., `week2`) for the evaluation.
- **`--SEQ_INFO`**: The sequence being evaluated (e.g., `s03`).
- **`--DO_PREPROC=False`**: Disables preprocessing of the data.

The table below presents the **HOTA** and **IDF1** scores for two different tracking algorithms:

| Tracker                                | Training Strategy  | HOTA Score | IDF1 Score |
|----------------------------------------|--------------------|------------|------------|
| Tracking by overlap (IoU=0.5)         | Fully Unfrozen     | 82.624     | 80.494     |
| Tracking by overlap (IoU=0.45)        | Fully Unfrozen     | 82.793     | 80.917     |
| Tracking by overlap (IoU=0.4)         | Fully Unfrozen     | 83.205     | 82.079     |
| Tracking by overlap (IoU=0.5)         | Backbone Frozen    | 75.846     | 70.764     |
| Tracking by overlap (IoU=0.45)        | Backbone Frozen    | 75.906     | 70.94      |
| Tracking by overlap (IoU=0.4)         | Backbone Frozen    | 75.747     | 71.574     |
| Tracking with Kalman Filter (KF, max_age=21, min_hits=2, IoU=0.2) | Fully Unfrozen     | 88.69 | 93.77 |
| Tracking with Kalman Filter (KF, max_age=21, min_hits=2, IoU=0.1) | Fully Unfrozen     | 88.66 | 93.70 |
| Tracking with Kalman Filter (KF, max_age=21, min_hits=3, IoU=0.1) | Fully Unfrozen     | 88.58 | 93.69 |
| Tracking with Kalman Filter (KF, max_age=21, min_hits=2, IoU=0.2) | Backbone Frozen    | 86.54 | 89.17 |
| Tracking with Kalman Filter (KF, max_age=21, min_hits=2, IoU=0.1) | Backbone Frozen    | 86.57 | 89.24 |
| Tracking with Kalman Filter (KF, max_age=21, min_hits=3, IoU=0.1) | Backbone Frozen    | 86.51 | 89.22 |


