import os
import shutil
from pathlib import Path
import yaml
from ultralytics import YOLO
import argparse
import numpy as np


def create_dir_structure(base_path: str, strategy="A"):
    """Create the directory structure for the dataset.

    Args:
        base_path (str): The base path where the dataset will be stored.
        strategy (str, optional): Either "A" for single split or "B" for K-fold. Defaults to "A".
    """
    if strategy == "A":
        directories = ["train/images", "train/labels", "val/images", "val/labels"]
        for dir_path in directories:
            Path(os.path.join(base_path, dir_path)).mkdir(parents=True, exist_ok=True)
    elif strategy == 'B':  # strategy B
        k = 4  # K=4 as per requirements
        for fold in range(k):
            fold_path = os.path.join(base_path, f'fold_{fold}')
            directories = ["train/images", "train/labels", "val/images", "val/labels"]
            for dir_path in directories:
                Path(os.path.join(fold_path, dir_path)).mkdir(parents=True, exist_ok=True)
    elif strategy == 'C':
        k = 4  # K=4 as per requirements
        for fold in range(k):
            fold_path = os.path.join(base_path, f'fold_random_{fold}')
            directories = ["train/images", "train/labels", "val/images", "val/labels"]
            for dir_path in directories:
                Path(os.path.join(fold_path, dir_path)).mkdir(parents=True, exist_ok=True)
        

def split_dataset(source_frames: str, source_labels: str, dest_base_path: str, train_ratio=0.25, strategy = 'A'):
    """Split the dataset into train and validation sets.

    Args:
        source_frames (str): Source path for the frames (contains frame_XXXX.jpg)
        source_labels (str): Source path for the labels (contains X.txt)
        dest_base_path (str): Destination path for the dataset
        train_ratio (float, optional): Train ratio in splitting. Defaults to 0.25.
    """
    # Get sorted list of frame numbers
    frame_files = sorted(os.listdir(source_frames))
    
    if strategy == 'A':
        split_idx = int(len(frame_files) * train_ratio)

        # Split frames into train and val
        train_frames = frame_files[:split_idx]
        val_frames = frame_files[split_idx:]

        # Copy frames and their corresponding labels for training set
        for frame in train_frames:
            frame_number = frame.split('_')[1].split('.')[0]
            
            # Copy image with the same filename as source
            shutil.copy2(
                os.path.join(source_frames, frame),
                os.path.join(dest_base_path, 'train/images', frame)
            )

            # Copy label with the same base name as the image
            label_file = f"{frame}"  # Use frame name for label
            label_file = label_file.replace('.jpg', '.txt')  # Replace extension
            source_label = os.path.join(source_labels, str(int(frame_number)) + '.txt')
            if os.path.exists(source_label):
                shutil.copy2(
                    source_label,
                    os.path.join(dest_base_path, 'train/labels', label_file)
                )

        # Copy frames and their corresponding labels for validation set
        for frame in val_frames:
            frame_number = frame.split('_')[1].split('.')[0]  # Gets XXXX from frame_XXXX.jpg
            
            # Copy image
            shutil.copy2(
                os.path.join(source_frames, frame),
                os.path.join(dest_base_path, 'val/images', frame)
            )

            # Copy label - Use the same naming format as training
            label_file = frame.replace('.jpg', '.txt')  # Same format as for training
            source_label = os.path.join(source_labels, str(int(frame_number)) + '.txt')
            if os.path.exists(source_label):
                shutil.copy2(
                    source_label,
                    os.path.join(dest_base_path, 'val/labels', label_file)
                )
    elif strategy == 'B': # strategy B
        k = 4  # K=4 as per requirements
        
        # Create indices for K-fold
        frame_indices = np.arange(len(frame_files))
        fold_size = len(frame_files) // k
        
        # Split data into K folds (fixed, not randomized)
        for fold in range(k):
            val_start = fold * fold_size
            val_end = (fold + 1) * fold_size if fold < k - 1 else len(frame_files)
            
            # Calculate validation indices for this fold
            train_indices = frame_indices[val_start:val_end]
            val_indices = np.setdiff1d(frame_indices, train_indices)
            
            fold_path = os.path.join(dest_base_path, f'fold_{fold}')
            
            # Process training set for this fold
            for idx in train_indices:
                frame = frame_files[idx]
                frame_number = frame.split('_')[1].split('.')[0]
                
                # Copy image
                shutil.copy2(
                    os.path.join(source_frames, frame),
                    os.path.join(fold_path, 'train/images', frame)
                )
                
                # Copy label
                label_file = frame.replace('.jpg', '.txt')
                source_label = os.path.join(source_labels, str(int(frame_number)) + '.txt')
                if os.path.exists(source_label):
                    shutil.copy2(
                        source_label,
                        os.path.join(fold_path, 'train/labels', label_file)
                    )
            
            # Process validation set for this fold
            for idx in val_indices:
                frame = frame_files[idx]
                frame_number = frame.split('_')[1].split('.')[0]
                
                # Copy image
                shutil.copy2(
                    os.path.join(source_frames, frame),
                    os.path.join(fold_path, 'val/images', frame)
                )
                
                # Copy label
                label_file = frame.replace('.jpg', '.txt')
                source_label = os.path.join(source_labels, str(int(frame_number)) + '.txt')
                if os.path.exists(source_label):
                    shutil.copy2(
                        source_label,
                        os.path.join(fold_path, 'val/labels', label_file)
                    )
    elif strategy == 'C':
        k = 4  # K=4 as per requirements
    
        # Create indices for K-fold with randomization
        frame_indices = np.arange(len(frame_files))
        np.random.shuffle(frame_indices)  # Randomize indices
        fold_size = len(frame_files) // k
        
        # Split data into K folds (randomized)
        for fold in range(k):
            val_start = fold * fold_size
            val_end = (fold + 1) * fold_size if fold < k - 1 else len(frame_files)
            
            # Calculate validation indices for this fold
            train_indices = frame_indices[val_start:val_end]
            val_indices = np.setdiff1d(frame_indices, train_indices)
            
            fold_path = os.path.join(dest_base_path, f'fold_random_{fold}')
            
            # Process training set for this fold (25%)
            for idx in train_indices:
                frame = frame_files[idx]
                frame_number = frame.split('_')[1].split('.')[0]
                
                # Copy image
                shutil.copy2(
                    os.path.join(source_frames, frame),
                    os.path.join(fold_path, 'train/images', frame)
                )
                
                # Copy label
                label_file = frame.replace('.jpg', '.txt')
                source_label = os.path.join(source_labels, str(int(frame_number)) + '.txt')
                if os.path.exists(source_label):
                    shutil.copy2(
                        source_label,
                        os.path.join(fold_path, 'train/labels', label_file)
                    )
            
            # Process validation set for this fold (75%)
            for idx in val_indices:
                frame = frame_files[idx]
                frame_number = frame.split('_')[1].split('.')[0]
                
                # Copy image
                shutil.copy2(
                    os.path.join(source_frames, frame),
                    os.path.join(fold_path, 'val/images', frame)
                )
                
                # Copy label
                label_file = frame.replace('.jpg', '.txt')
                source_label = os.path.join(source_labels, str(int(frame_number)) + '.txt')
                if os.path.exists(source_label):
                    shutil.copy2(
                        source_label,
                        os.path.join(fold_path, 'val/labels', label_file)
                    )
            
def create_dataset_yml(base_path: str, num_classes: int = 1, class_names: list = ['car'], strategy= 'A', fold=None):
    """Create the dataset YAML file.

    Args:
        dest_base_path (str): Destination path for the dataset.
        num_classes (int, optional): Number of classes. Defaults to 1.
        class_names (list, optional): List of class names. Defaults to ['car'].
    """
    if strategy == 'A':
        dataset_yaml = {
            'path': base_path,
            'train': 'train/images',
            'val': 'val/images',
            'nc': num_classes,
            'names': {i: name for i, name in enumerate(class_names)}
        }
        yaml_path = os.path.join(base_path, 'dataset.yaml')
    elif strategy == 'B': # Strategy B    
        fold_path = os.path.join(base_path, f'fold_{fold}')
        dataset_yaml = {
            'path': fold_path,
            'train': 'train/images',
            'val': 'val/images',
            'nc': num_classes,
            'names': {i: name for i, name in enumerate(class_names)}
        }
        yaml_path = os.path.join(base_path, f'dataset_fold_{fold}.yaml')
    elif strategy == 'C':
        fold_path = os.path.join(base_path, f'fold_random_{fold}')
        dataset_yaml = {
            'path': fold_path,
            'train': 'train/images',
            'val': 'val/images',
            'nc': num_classes,
            'names': {i: name for i, name in enumerate(class_names)}
        }
        yaml_path = os.path.join(base_path, f'dataset_fold_random_{fold}.yaml')
        
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_yaml, f)
        
    return yaml_path

def evaluate_kfold_results(base_path: str, k=4):
    """
    Evaluate results from K-Fold cross-validation using CSV files.
    
    Args:
        base_path (str): Base path where training outputs are stored
        k (int, optional): Number of folds. Defaults to 4.
        
    Returns:
        dict: Aggregated metrics across all folds
    """
    import os
    import pandas as pd
    import numpy as np
    
    all_metrics = []
    
    for fold in range(k):
        results_path = os.path.join(base_path, f'yolo_train_fold_{fold}', 'results.csv')
        if os.path.exists(results_path):
            # Read the CSV file
            df = pd.read_csv(results_path)
            
            # Get the final epoch results (last row)
            final_epoch = df.iloc[-1].to_dict()
            final_epoch['fold'] = fold
            all_metrics.append(final_epoch)
    
    # Calculate average metrics
    if all_metrics:
        # Find all metric columns (those that start with 'metrics/')
        metrics_keys = [key for key in all_metrics[0].keys() if isinstance(key, str) and key.startswith('metrics/')]
        avg_metrics = {}
        
        for key in metrics_keys:
            values = [metrics[key] for metrics in all_metrics if key in metrics]
            avg_metrics[f'avg_{key}'] = np.mean(values)
            avg_metrics[f'std_{key}'] = np.std(values)
        
        # Print summary
        print("=== K-Fold Cross-Validation Results ===")
        for key, value in avg_metrics.items():
            if key.startswith('avg_'):
                metric_name = key[4:]  # Remove 'avg_' prefix
                std_key = f'std_{metric_name}'
                if std_key in avg_metrics:
                    print(f"{metric_name}: {value:.4f} ± {avg_metrics[std_key]:.4f}")
                else:
                    print(f"{metric_name}: {value:.4f}")
        
        return avg_metrics
    else:
        print("No results found for K-fold evaluation.")
        return {}


def train_yolo(yaml_path: Path, epochs: int=50, batch_size: int=8, imgsz: int=640, fold=None, strategy = 'B'):
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Dataset YAML file not found: {yaml_path}")
        
    # Load and validate YAML
    with open(yaml_path, 'r') as f:
        yaml_data = yaml.safe_load(f)
        
    # Check required paths exist
    train_path = os.path.join(yaml_data['path'], yaml_data['train'])
    val_path = os.path.join(yaml_data['path'], yaml_data['val'])
    
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError(f"Training or validation directory not found")
    
    # Load the model
    model = YOLO('yolo11x.pt')
    
    # Set project name based on fold
    if strategy == 'B':
        project_name = "yolo_train" if fold is None else f"yolo_train_fold_{fold}"
    elif strategy == 'C':
        project_name = "yolo_train" if fold is None else f"yolo_train_fold_random_{fold}"
    
    # Train the model
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        device='cuda',
        patience=20,  # Early stopping patience,
        project=project_name
    )
    
    return results


def main():
    # Define paths
    BASE_DIR = '/ghome/c3mcv02/mcv-c6-2025-team1'
    SOURCE_FRAMES = f'{BASE_DIR}/data/train/frames'
    SOURCE_LABELS = f'{BASE_DIR}/data/train/labels'
    DATASET_PATH = f'{BASE_DIR}/week2/dataset'
    
    # TODO: Change strategy
    parser = argparse.ArgumentParser(description='Train YOLO with different splitting strategies')
    parser.add_argument('--strategy', type=str, choices=['A', 'B', 'C'], required=True,
                        help='Splitting strategy: A (simple split), B (K-fold), C (K-Fold, random)')
    args = parser.parse_args()
    strategy = args.strategy
    print(f"Using strategy {strategy}")
    
    # Create directory structure
    create_dir_structure(DATASET_PATH, strategy=strategy)
    
    # Split and organize dataset
    split_dataset(SOURCE_FRAMES, SOURCE_LABELS, DATASET_PATH, train_ratio=0.25, strategy=strategy)
    
    if strategy == 'A':
        # Create dataset YAML file
        yaml_path = create_dataset_yml(DATASET_PATH)
        
        # sanity checks before training
        print(f"Checking paths...")
        print(f"Train images path: {os.path.join(DATASET_PATH, 'train/images')}")
        print(f"Val images path: {os.path.join(DATASET_PATH, 'val/images')}")
        print(f"Train labels path: {os.path.join(DATASET_PATH, 'train/labels')}") 
        print(f"Val labels path: {os.path.join(DATASET_PATH, 'val/labels')}") 
        print("Directory contents:")
        for path in ['train/images', 'train/labels', 'val/images', 'val/labels']:
            full_path = os.path.join(DATASET_PATH, path)
            files = os.listdir(full_path) if os.path.exists(full_path) else []
            print(f"{path}: {len(files)} files")
        
        # Train the model
        results = train_yolo(
            yaml_path,
            epochs=50,
            batch_size=8,
            imgsz=640
        )
        
        print("Training completed!")
        print(f"Results saved in: {results.save_dir}")
    elif strategy == 'B' or strategy == 'C':  # Strategy B - K-fold
        k = 4  # K=4 as per requirements
        all_results = []
        
        for fold in range(k):
            print(f"Processing fold {fold+1}/{k}")
            
            # Create dataset YAML file for this fold
            yaml_path = create_dataset_yml(DATASET_PATH, strategy=strategy, fold=fold)
            
            # Sanity checks before training
            fold_path = os.path.join(DATASET_PATH, f'fold_{fold}')
            print(f"Checking paths for fold {fold}...")
            print(f"Train images path: {os.path.join(fold_path, 'train/images')}")
            print(f"Val images path: {os.path.join(fold_path, 'val/images')}")
            print(f"Train labels path: {os.path.join(fold_path, 'train/labels')}") 
            print(f"Val labels path: {os.path.join(fold_path, 'val/labels')}") 
            print("Directory contents:")
            for path in ['train/images', 'train/labels', 'val/images', 'val/labels']:
                full_path = os.path.join(fold_path, path)
                files = os.listdir(full_path) if os.path.exists(full_path) else []
                print(f"{path}: {len(files)} files")
            
            # Train the model for this fold
            results = train_yolo(
                yaml_path,
                epochs=50,
                batch_size=8,
                imgsz=640,
                fold=fold,
                strategy=strategy
            )
            
            all_results.append(results)
            print(f"Fold {fold+1}/{k} training completed!")
            print(f"Results saved in: {results.save_dir}")
        
        # Evaluate K-fold results
        evaluate_kfold_results(BASE_DIR, k=k)
    else:
        raise NotImplementedError()

if __name__ == "__main__":
    main()