import pandas as pd
import numpy as np 
import os
import argparse


def evaluate_kfold_results(base_path: str, k=4, is_random: bool = False):
    """
    Evaluate results from K-Fold cross-validation using CSV files.
    
    Args:
        base_path (str): Base path where training outputs are stored
        k (int, optional): Number of folds. Defaults to 4.
        
    Returns:
        dict: Aggregated metrics across all folds
    """
    all_metrics = []
    
    for fold in range(k):
        if not is_random:
            results_path = os.path.join(base_path, f'yolo_train_fold_{fold}', 'train/results.csv')
        else:
            results_path = os.path.join(base_path, f'yolo_train_fold_random_{fold}', 'train/results.csv')
        
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
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--base_path", type=str, help="Path to the fold results.", required=True)
    parser.add_argument("--is_random", action="store_true", help="Wether is random fold or fixed.")
    args = parser.parse_args()
    
    # Get the base_path from arguments
    base_path = args.base_path
    is_random = args.is_random
    
    evaluate_kfold_results(base_path, k=4, is_random=is_random)
    