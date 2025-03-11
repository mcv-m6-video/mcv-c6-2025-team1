<h2 align="center">WEEK 3: Tasks</h2>

## Table of Contents

- [Project Structure W3](#project-structure-w3)
- [Task 1: Optical flow](#task-1-optical-flow)  
    - [Task 1.1: Off-the-shelf](#task-11-off-the-shelf)  
    - [Task 1.2: Improve tracking with optical flow](#task-12-improve-tracking-with-optical-flow)  

- [Task 2: Multi-Target Single-Camera tracking](#task-2-multi-target-single-camera-tracking)  
    - [Task 2.1: Evaluate your best tracking algorithm in SEQ01 of AI City Challenge](#task-21-evaluate-your-best-tracking-algorithm-in-seq01-of-ai-city-challenge)  
    - [Task 2.2: Evaluate your best tracking algorithm in SEQ03 of AI City Challenge](#task-22-evaluate-your-best-tracking-algorithm-in-seq03-of-ai-city-challenge)  


### Project Structure W3

    week3/
    ├── src/
    │   ├── optical_flow/
    │   │   ├── of.py
    │   │   ├── utils.py
    │   │   └── ...
    │   ├── detection.py
    │   ├── utils.py
    │   └── ...

## Task 1: Optical Flow

Optical flow is a computer vision technique used to estimate motion between consecutive frames of a video or image sequence by analyzing pixel displacements. It helps in tracking objects, motion estimation, and video processing tasks. 

### Task 1.1: Off-the-Shelf
These are the off-the-shelf optical flow methods that we have tested in the Sequence 45 of the KITTI 2012 dataset. We have utilized the implementation from [PTLFlow](https://github.com/hmorimitsu/ptlflow) for testing the methods. 

**1. PyFlow**: A Python implementation of Coarse2Fine Optical Flow, a variational approach for dense motion estimation that refines the motion field at multiple scales.

**2. MemFlow**: Uses a memory module for real-time optical flow estimation and introduces resolution-adaptive re-scaling in attention computation, allowing for improved accuracy in dynamic scenes.

**3. RAPIDFlow**: Implements a recurrent encoder-decoder architecture, utilizing efficient 1D layers to generate feature pyramids for multi-scale motion estimation, making it suitable for fast-moving objects.

**4. RPKNet**: Employs a spatial recurrent encoding structure with PKConv layers that extract discriminative multi-scale features while using a Separable Large Kernel module, which relies only on 1D convolutions for computational efficiency.

**5. DICL-Flow**: Learns matching costs from concatenated features to improve optical flow estimation and introduces a displacement-aware projection layer to refine motion correlation between different motion hypotheses.

**6. DIP**: An end-to-end deep learning method based on Patchmatch, designed for high-resolution optical flow prediction. It introduces a novel inverse propagation module to enhance motion estimation in complex scenes.

To run any of the above methods, use the following command:

```bash
python optical_flow/of.py -m <METHOD> -gt <GT_PATH> -im1 <PATH_TO_FIRST_IMG> -im2 <PATH_TO_SECOND_IMG> 
```

### Task 1.2: Improve Tracking with Optical Flow
This script performs multi-object tracking using BoxMOT trackers with offline detections and optical flow integration. The tracker processes video frames, combining pre-computed detections with optical flow estimates to generate tracked object trajectories.

To run the tracker, use the following command:

```bash
python -m src.tracking.tracking_boxmot -d /path/to/detections.txt -v /path/to/video.mp4 -ov /path/to/output_video.mp4 -o /path/to/output_tracks.txt
```

Required arguments are the detection file path `-d`, input video path `-v`, output video path `-ov`, and output tracking results path `-o`. The detection file should follow the format: `<frame_number> <x> <y> <width> <height>` per line.

Additional options allow customization of the tracking process. The tracking method can be specified with `-m` (defaults to "deepocsort"), with support for deepocsort, botsort, strongsort, ocsort, and bytetrack. The optical flow model can be selected using `--of_model` (defaults to "rpknet"), supporting various models like pyflow, diclflow, memflow, rapidflow, rpknet, and dip.

Fine-tuning parameters include the IoU threshold `--iou_threshold`, detection-prediction fusion weight `--alpha`, optical flow prediction method `--pred_method`, and Gaussian weighting sigma `--sigma`. For GPU acceleration, use `--device cuda` and optionally enable half-precision inference with `--half`.

In order to pass additional configuration settings for the SORT variation algorithm, you will have to set up a configuration file similar to the ones in [here](https://github.com/mcv-m6-video/mcv-c6-2025-team1/tree/main/week3/configs). The pass the configuration to the script, use the option `-c`.

The script outputs both an annotated video showing tracked objects with their IDs and a text file containing tracking results in MOTChallenge format: `<frame_id>,<track_id>,<x>,<y>,<w>,<h>,1,-1,-1,-1`. 

#### Using SORT with Optical Flow Integration

The SORT (Simple Online and Realtime Tracking) implementation extends the base tracker by incorporating optical flow information to improve tracking stability. Run it using:

```bash
python -m src.tracking.tracking_kf -d /path/to/detections.txt -v /path/to/video.mp4 -ov /path/to/output_video.mp4 -o /path/to/output_tracks.txt
```

SORT-specific parameters can be tuned via:
```bash
--max_age 21        # Maximum frames to keep alive a track without associated detections
--min_hit 2         # Minimum hits to start a track
--iou_threshold 0.2 # Minimum IoU for match
```

The script integrates SORT's Kalman filtering with optical flow estimation, using weighted Gaussian, mean, median or max averaging for improved motion prediction. This hybrid approach helps maintain tracking consistency through occlusions and missed detections. 

## Task 2: Multi-Target Single-Camera Tracking
In this task, we evaluate the performance of two different tracking algorithms in the AI City Challenge, specifically focusing on SEQ01 and SEQ03. We will assess the effectiveness of both the tracking algorithm from Week 2 (SORT combined with the Kalman Filter) and the best algorithm developed this week

### Task 2.1: Evaluate Your Best Tracking Algorithm in SEQ01 of AI City Challenge

### Task 2.2: Evaluate Your Best Tracking Algorithm in SEQ03 of AI City Challenge

### Evaluation Method:

In this task, we will evaluate the performance of the best tracking algorithm using the **TrackEval** framework. We will calculate two important metrics:

- **HOTA (Higher Order Tracking Accuracy)**: This metric captures both the quality of tracking (how well objects are tracked) and the quality of object detection (how well objects are detected).
- **IDF1 (Identification F1 Score)**: This metric measures the ability of the algorithm to correctly identify and associate objects across frames.

### Evaluation Command Explanation

The command to evaluate your tracking algorithm is as follows:

```bash
python multi_target_tracking/TrackEval/scripts/run_mot_challenge.py --GT_FOLDER /ghome/c5mcv01/mcv-c6-2025-team1/week3/src/multi_target_tracking/TrackEval/data/gt/mot_challenge --TRACKERS_FOLDER /ghome/c5mcv01/mcv-c6-2025-team1/week3/src/multi_target_tracking/TrackEval/data/trackers/mot_challenge --BENCHMARK week3 --SEQ_INFO c001 c002 c003 c004 c005 --DO_PREPROC=False
```

