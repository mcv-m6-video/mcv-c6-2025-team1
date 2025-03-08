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
python optical_flow/of.py --m <method>
```

### Task 1.2: Improve Tracking with Optical Flow

## Task 2: Multi-Target Single-Camera Tracking

### Task 2.1: Evaluate Your Best Tracking Algorithm in SEQ01 of AI City Challenge

### Task 2.2: Evaluate Your Best Tracking Algorithm in SEQ03 of AI City Challenge

