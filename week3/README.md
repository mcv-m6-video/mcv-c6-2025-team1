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

