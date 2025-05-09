import os
import cv2
import numpy as np
import argparse
import pickle

from PIL import Image
from pathlib import Path
from typing import Dict, List
from sklearn.metrics.pairwise import cosine_similarity

from src.algoritm_features.classes import Detection
from src.algoritm_features.feature_extractor_finetuned import FeatureExtractor as ResNeXtExtractor
from src.algoritm_features.feature_extractor import FeatureExtractor as ResNetExtractor
from scipy.spatial.distance import pdist, squareform


def re_identify_objects(detections_by_camera, **kwargs):
    """
    Re-identify objects across multiple cameras by matching features.

    Args:
        detections_by_camera: Dictionary mapping camera IDs to lists of Detection objects

    Returns:
        Dictionary mapping camera IDs to lists of Detection objects with updated global IDs
    """
    print("Starting re-identification across cameras...")

    # Create a global ID tracker
    next_global_id = 1
    global_id_mapping = {}  # Maps (camera_id, track_id) -> global_id

    # Store all detections with features for matching
    all_detections_with_features = []

    # First pass: collect all detections with features
    for camera_id, detections in detections_by_camera.items():
        for detection in detections:
            if detection.features:
                # Use the first feature vector for matching (can be extended to use multiple)
                detection.global_feature = detection.features[0]
                all_detections_with_features.append(detection)
    print(f"Found {len(all_detections_with_features)} detections with features")

    # Group detections by track_id within each camera
    camera_track_groups = {}
    for camera_id, detections in detections_by_camera.items():
        camera_track_groups[camera_id] = {}
        for detection in detections:
            if detection.track_id not in camera_track_groups[camera_id]:
                camera_track_groups[camera_id][detection.track_id] = []
            camera_track_groups[camera_id][detection.track_id].append(detection)

    # For each camera and track_id, compute an average feature
    # TODO: Improve this part for averaging features
    track_representatives = []
    for camera_id, track_dict in camera_track_groups.items():
        for track_id, track_detections in track_dict.items():
            # Filter detections with features
            detections_with_features = [d for d in track_detections if hasattr(d, 'global_feature')]

            if not detections_with_features:
                continue

            # Compute average feature vector
            feature_vectors = [d.global_feature for d in detections_with_features]
            avg_feature = np.mean(feature_vectors, axis=0)

            # Compute average timestamp for the track
            timestamp_vector = [d.timestamp for d in detections_with_features]
            avg_timestamp = np.mean(timestamp_vector, axis=0)

            # Use middle detection as representative
            middle_idx = len(detections_with_features) // 2
            representative = detections_with_features[middle_idx]
            representative.avg_feature = avg_feature

            track_representatives.append({
                'camera_id': camera_id,
                'track_id': track_id,
                "timestamp": avg_timestamp,
                'representative': representative
            })
    print(f"Created {len(track_representatives)} track representatives")

    # Create feature matrix for efficient computation
    feature_matrix = np.array([rep['representative'].avg_feature for rep in track_representatives])
    if kwargs.get("similarity_type") == "similarity":
        similarities = cosine_similarity(feature_matrix)
    elif kwargs.get("similarity_type") == "distance":
        dists = squareform(pdist(feature_matrix, metric='cityblock'))
        min_vals = dists.min(axis=1, keepdims=True)
        max_vals = dists.max(axis=1, keepdims=True)
        distances = (dists - min_vals) / (max_vals - min_vals)
    else:
        raise ValueError("Invalid similarity type")


    # Process similarities to group tracks
    processed = set()
    track_groups = []

    print("Grouping similar tracks across cameras...")
    for i in range(len(track_representatives)):
        if i in processed:
            continue

        # Start a new group with this track
        current_group = [i]
        processed.add(i)

        # Group potential matches by camera to find the best match per camera
        camera_best_matches = {}  # camera_id -> (index, similarity, time_diff)

        # Find all matches above threshold
        for j in range(len(track_representatives)):
            if j in processed or i == j:
                continue

            # Don't match within the same camera - different tracks in same camera should remain separate
            if track_representatives[i]['camera_id'] == track_representatives[j]['camera_id']:
                continue

            # Calculate time difference
            time_diff = abs(track_representatives[i]['timestamp'] - track_representatives[j]['timestamp'])

            # Prune by timestamp difference
            if time_diff > kwargs.get("timestamp_threshold", 10):
                continue

            if kwargs.get("similarity_type") == "similarity":
                if similarities[i, j] > kwargs.get("similarity_threshold", 0.9):
                    camera_j = track_representatives[j]['camera_id']

                    # If this is the first match for this camera, or it has a smaller time difference
                    if camera_j not in camera_best_matches or time_diff < camera_best_matches[camera_j][2]:
                        camera_best_matches[camera_j] = (j, similarities[i, j], time_diff)
            elif kwargs.get("similarity_type") == "distance":
                if distances[i, j] < kwargs.get("similarity_threshold", 0.9):
                    camera_j = track_representatives[j]['camera_id']

                    # If this is the first match for this camera, or it has a smaller time difference
                    if camera_j not in camera_best_matches or time_diff < camera_best_matches[camera_j][2]:
                        camera_best_matches[camera_j] = (j, distances[i, j], time_diff)
            else:
                raise ValueError("Invalid similarity type")

        # Add only the best match from each camera to the current group
        for camera_id, (j, similarity, time_diff) in camera_best_matches.items():
            print(f"  Matched tracks {track_representatives[i]['track_id']} / {track_representatives[i]['timestamp']} at camera {track_representatives[i]['camera_id']} "
                  f"with tracks {track_representatives[j]['track_id']} / {track_representatives[j]['timestamp']} at camera {camera_id} with similarity {similarity}")
            current_group.append(j)
            processed.add(j)

        if len(current_group) > 1:
            track_groups.append(current_group)

    # Assign global IDs to each group
    for group in track_groups:
        global_id = next_global_id
        next_global_id += 1

        for idx in group:
            camera_id = track_representatives[idx]['camera_id']
            track_id = track_representatives[idx]['track_id']
            global_id_mapping[(camera_id, track_id)] = global_id

    # TODO: Handle tracks that weren't matched (assign global_id = -1?)
    # In here maybe we can remove the tracks that weren't matched
    for camera_id, track_dict in camera_track_groups.items():
        for track_id in track_dict:
            if (camera_id, track_id) not in global_id_mapping:
                print("Track not matched:", camera_id, track_id)
                global_id_mapping[(camera_id, track_id)] = -1
                # next_global_id += 1

    # Update all detections with global IDs
    for camera_id, detections in detections_by_camera.items():
        for detection in detections:
            if (camera_id, detection.track_id) in global_id_mapping:
                detection.global_id = global_id_mapping[(camera_id, detection.track_id)]
            else:
                # This shouldn't happen if all track IDs are properly processed
                print(f"Warning: No global ID found for detection in camera {camera_id} with track ID {detection.track_id}")
                detection.global_id = -1
    print(f"Re-identification complete. Assigned {next_global_id-1} global IDs.")

    # Optional: Print statistics
    camera_counts = {camera_id: len(set(d.global_id for d in detections))
                     for camera_id, detections in detections_by_camera.items()}
    print("Objects detected per camera (after re-ID):")
    for camera_id, count in camera_counts.items():
        print(f"  Camera {camera_id}: {count} unique objects")
    return detections_by_camera


def load_detections_and_extract_features(
        detections_folder: str,
        videos_folder: str,
        camera_offsets: Dict[str, float] = None,
        camera_ids: List[str] = None,
        feature_extractor: str = "resnext"
) -> Dict[str, List[Detection]]:
    """
    Load detections from txt files and extract features from video frames.

    Args:
        detections_folder: Path to folder containing detection files
        videos_folder: Path to folder containing video files
        camera_ids: List of camera IDs to process (if None, process all)

    Returns:
        Dictionary mapping camera IDs to lists of Detection objects with features
    """
    # Initialize feature extractor
    if feature_extractor == "resnext":
        FeatureExtractor = ResNeXtExtractor
    elif feature_extractor == "resnet":
        FeatureExtractor = ResNetExtractor
    feature_extractor = FeatureExtractor()

    # Dictionary to store detections by camera ID
    all_detections = {}

    # Get all detection files
    detection_files = list(Path(detections_folder).glob("*.txt"))

    for file_path in detection_files:
        # Extract camera ID from filename
        camera_id = file_path.stem.split('.')[0]

        # Skip if not in requested camera IDs
        if camera_ids and camera_id not in camera_ids:
            continue
        print(f"Processing detections for camera {camera_id}")

        # Load video
        video_path = os.path.join(videos_folder, f"{camera_id}.avi")
        if not os.path.exists(video_path):
            print(f"Video file not found: {video_path}")
            continue

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Could not open video: {video_path}")
            continue

        # Get FPS directly from the video
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Camera {camera_id} FPS: {fps}")

        # Read detections
        detections = []
        with open(file_path, 'r') as f:
            lines = f.readlines()

        # Group detections by frame to efficiently process the video
        frame_to_detections = {}
        for line in lines:
            # Get the frame number and timestamp
            parts = line.strip().split(',')
            frame_num = int(parts[0])
            timestamp = camera_offsets[camera_id] + (frame_num / fps)

            # Create Detection object
            detection = Detection.from_string(line, timestamp, camera_id)
            frame_num = detection.frame_num
            if frame_num not in frame_to_detections:
                frame_to_detections[frame_num] = []
            frame_to_detections[frame_num].append(detection)
            detections.append(detection)

        # Process video frame by frame
        for frame_num in sorted(frame_to_detections.keys()):
            # Set video to the correct frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)  # 0-based indexing
            ret, frame = cap.read()

            if not ret:
                print(f"Could not read frame {frame_num} from video {video_path}")
                continue

            if frame_num % 100 == 0:
                print(f"Processing frame {frame_num}")

            # Process all detections for this frame
            for detection in frame_to_detections[frame_num]:
                x, y, width, height = detection.get_bbox()

                # Extract the region of interest
                try:
                    # Ensure coordinates are within frame boundaries
                    x, y = max(0, x), max(0, y)
                    width = min(width, frame.shape[1] - x)
                    height = min(height, frame.shape[0] - y)

                    if width <= 0 or height <= 0:
                        continue

                    # Extract bbox region
                    bbox_region = frame[y:y+height, x:x+width]
                    bbox_region_rgb = cv2.cvtColor(bbox_region, cv2.COLOR_BGR2RGB)

                    # Extract features from the region
                    pil_image = Image.fromarray(bbox_region_rgb)
                    feature_vector = feature_extractor.extract_features(pil_image)

                    # Add features to detection
                    detection.add_feature(feature_vector)

                except Exception as e:
                    print(f"Error extracting features for detection in frame {frame_num}: {e}")
        # Close video
        cap.release()

        # Store detections for this camera
        all_detections[camera_id] = detections
        print(f"Processed {len(detections)} detections for camera {camera_id}")
        print("-" * 50)
    return all_detections


def save_detections_with_global_ids(detections_by_camera, output_folder):
    """
    Save detections with global IDs to text files, one file per camera.

    Args:
        detections_by_camera: Dictionary mapping camera IDs to lists of Detection objects
        output_folder: Path to the folder where output files will be saved
    """
    import os

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    print(f"Saving detections with global IDs to {output_folder}...")

    # Process each camera
    for camera_id, detections in detections_by_camera.items():
        output_file = os.path.join(output_folder, f"{camera_id}.txt")

        # Sort detections by frame number
        sorted_detections = sorted(detections, key=lambda d: d.frame_num)

        # Count valid detections (those with positive global IDs)
        valid_detections = [d for d in sorted_detections if hasattr(d, 'global_id') and d.global_id > 0]

        with open(output_file, 'w') as f:
            for detection in valid_detections:
                # Format: FRAME_ID, GLOBAL_TRACK_ID, X, Y, WIDTH, HEIGHT, confidence, -1, -1, -1
                line = f"{detection.frame_num},{detection.global_id},{detection.x},{detection.y},"
                line += f"{detection.width},{detection.height},{detection.confidence},-1,-1,-1\n"
                f.write(line)
        print(f"  Saved {len(sorted_detections)} detections to {output_file}")
    print("All detections saved successfully.")


def get_camera_ids_from_files(detections_folder):
    """Extract camera IDs from detection file names in the folder."""
    detection_files = list(Path(detections_folder).glob("*.txt"))
    camera_ids = [file_path.stem.split('.')[0] for file_path in detection_files]
    return camera_ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-identification algorithm")
    parser.add_argument("--detections_folder", type=str, required=True,
                        help="Path to folder containing detection files")
    parser.add_argument("--videos_folder", type=str, required=True,
                        help="Path to folder containing video files")
    parser.add_argument("--output_folder", type=str, required=True,
                        help="Path to folder where output files will be saved")
    parser.add_argument("--camera_offsets", type=str, default=None, required=True,
                        help="Path to the TXT file with camera timestamp offsets")
    parser.add_argument("--similarity_threshold", type=float, default=0.90, required=False,
                        help="Threshold for cosine similarity")
    parser.add_argument("--timestamp_threshold", type=int, default=10, required=False,
                        help="Threshold for timestamp difference")
    parser.add_argument("--pickle_file", type=str, default=None, required=False,
                   help="Path to pickle file with detections_by_camera dictionary")
    parser.add_argument("--similarity_type", type=str, default=None, required=False, choices=["similarity", "distance"],
                   help="Cosine similarity or manhattan distance measure ('similarity' or 'distance')")
    parser.add_argument("--feature_extractor", type=str, default="resnext", required=False, choices=["resnet", "resnext"],
                        help="Feature extractor model (resnet or resnext)")
    args = parser.parse_args()

    # Get the camera offsets from a TXT file
    camera_offsets = {}
    with open(args.camera_offsets, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                camera_id = parts[0]
                offset = float(parts[1])
                camera_offsets[camera_id] = offset
    print(f"Loaded {len(camera_offsets)} camera with offsets: {camera_offsets}")

    # Get the camera IDs from the detection files
    camera_ids = get_camera_ids_from_files(args.detections_folder)
    print(f"Found camera IDs: {camera_ids}")

    # Load detections and extract features for all cameras
    if args.pickle_file and os.path.exists(args.pickle_file):
        print(f"Loading detections from pickle file: {args.pickle_file}")
        with open(args.pickle_file, 'rb') as f:
            detections_by_camera = pickle.load(f)
    else:
        detections_by_camera = load_detections_and_extract_features(
            detections_folder=args.detections_folder,
            videos_folder=args.videos_folder,
            camera_offsets=camera_offsets,
            camera_ids=camera_ids,
            feature_extractor=args.feature_extractor
        )
        
        # Save the detections to pickle for future use
        pickle_path = os.path.join(args.output_folder, 'detections_by_camera.pkl')
        os.makedirs(args.output_folder, exist_ok=True)
        print(f"Saving detections to pickle file: {pickle_path}")
        with open(pickle_path, 'wb') as f:
            pickle.dump(detections_by_camera, f)

    # Define re-identification parameters
    kwargs = {
		"similarity_threshold": args.similarity_threshold,
		"timestamp_threshold": args.timestamp_threshold,
        "similarity_type": args.similarity_type
	}

    # Perform re-identification
    detections_by_camera = re_identify_objects(detections_by_camera, **kwargs)

    # Save detections with global IDs
    save_detections_with_global_ids(detections_by_camera, args.output_folder)
