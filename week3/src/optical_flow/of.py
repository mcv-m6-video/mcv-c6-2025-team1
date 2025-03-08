import numpy as np
import cv2
import time
import argparse
import pyflow
import ptlflow

from plots import hsv_plot, visualize_arrow
from utils import calculate_msen, calculate_pepn, read_png_file
from ptlflow.utils.io_adapter import IOAdapter


def compute_flow(img1: np.ndarray, img2: np.ndarray, model: str, params: list) -> np.ndarray:
    """Computes the optical flow based on the chosen model and parameters

    Args:
        img1 (np.ndarray): First image array
        img2 (np.ndarray): Second image array
        model (str): Model name
        params (list): List of parameters for the model

    Raises:
        NotImplementedError: Raises if model is not implemented.

    Returns:
        np.ndarray: Stack of horizontal and vertical flow.
    """
    if model == "pyflow":
        alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations, nSORIterations, colType = params
        u, v, _ = pyflow.coarse2fine_flow(
            img1, img2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations, nSORIterations, colType
        )
        return np.dstack((u, v))
    elif model == "diclflow" or model == "memflow" or model=="rapidflow" or model=="rpknet" or model=="dip":
        if model == "diclflow":
            model_ptlflow = ptlflow.get_model('dicl', ckpt_path='things')  # Load RAFT model with appropriate checkpoint
        elif model == "memflow":
            model_ptlflow = ptlflow.get_model('memflow', ckpt_path='things') 
        elif model == "rapidflow":
            model_ptlflow = ptlflow.get_model('rapidflow', ckpt_path='things')
        elif model == "rpknet":
            model_ptlflow = ptlflow.get_model('rpknet', ckpt_path='things')
        elif model == "dip":
            model_ptlflow = ptlflow.get_model('dip', ckpt_path='things')
    
            
        
        # Preprocess images for the RAFT model
        img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)  # RAFT expects RGB images
        img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        
        io_adapter = IOAdapter(model_ptlflow, img1_rgb.shape[:2])
        inputs = io_adapter.prepare_inputs([img1_rgb, img2_rgb])
        
        # Forward the inputs through the model
        predictions = model_ptlflow(inputs)
        
        flow = predictions['flows'][0, 0]  # Remove batch and sequence dimensions
        flow = flow.permute(1, 2, 0)  # Change from CHW to HWC shape
        
        u = flow[:, :, 0].detach().numpy()  # Extract the horizontal flow component (u)
        v = flow[:, :, 1].detach().numpy()  # Extract the vertical flow component (v)
        
        return np.dstack((u, v))
    else:
        raise NotImplementedError(f"Model '{model}' is not supported yet.")


def run_model(model: str, im1_path: str, img1: np.ndarray, img2: np.ndarray, gt: np.ndarray, params: list) -> tuple[float, float, float]:
    """Runs the selected optical flow model, computes MSEN, PEPN and inference time

    Args:
        model (str): Model name
        im1_path (str): Image file path
        img1 (np.ndarray): First image array
        img2 (np.ndarray): Second image array
        gt (np.ndarray): Ground Truth array.
        params (list): List of parameters

    Returns:
        tuple: Tuple of metrics (MSEN, PEPN) and inference time.
    """
    # Compute flow
    start_time = time.time()
    flow = compute_flow(img1, img2, model, params)
    inference_time = time.time() - start_time
    
    # Calculate MSEN and PEPN
    msen = calculate_msen(gt, flow)
    pepn = calculate_pepn(gt, flow)

    # Visualize and save the results
    hsv_plot(flow, filename=model)
    hsv_plot(gt, filename="GT")
    
    visualize_arrow(im1_path, gt, filename="GT")
    visualize_arrow(im1_path, flow, filename=model)
    
    return msen, pepn, inference_time


def main(args):
    # Fixed image paths and ground truth flow path
    gt_path = args.gt_path
    im1_path = args.img1_path
    im2_path = args.img2_path
    gt = read_png_file(gt_path)

    # Define model parameters (example for PyFlow, modify as needed)
    if args.m == "pyflow":
        img1 = cv2.imread(im1_path, cv2.IMREAD_GRAYSCALE)  
        img2 = cv2.imread(im2_path, cv2.IMREAD_GRAYSCALE)
        img1 = np.atleast_3d(img1.astype(float) / 255.0)
        img2 = np.atleast_3d(img2.astype(float) / 255.0)
        params = [0.012, 0.75, 20, 1, 1, 30, 1]  # Example: alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations, nSORIterations, colType
    elif args.m == "diclflow" or args.m == "memflow" or args.m == "rapidflow" or args.m =="rpknet" or args.m == "dip":
        img1 = cv2.imread(im1_path, cv2.IMREAD_GRAYSCALE)  
        img2 = cv2.imread(im2_path, cv2.IMREAD_GRAYSCALE)
        params = []
    else:
        print(f"Model {args.m} not implemented yet.")
        return

    # Run the model and compute metrics
    msen, pepn, inference_time = run_model(args.m, im1_path, img1, img2, gt, params)

    # Print results
    print(f"\nModel: {args.m}")
    print(f"MSEN: {msen:.4f}")
    print(f"PEPN: {pepn * 100:.2f}%")
    print(f"Inference Time: {inference_time:.2f} seconds")


if __name__ == "__main__":
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description="Run optical flow models and compute metrics.")
    parser.add_argument('-m', '--model', type=str, required=True, help="Optical flow model to use (e.g., 'pyflow', 'raft','memflow').")
    parser.add_argument('-gt', '--gt_path', type=str, required=True, default="/ghome/c5mcv01/mcv-c6-2025-team1/data/data_stereo_flow/training/flow_noc/000045_10.png", help="Path to the Ground Truth stereo flow.")
    parser.add_argument('-im1', '--img1_path', type=str, required=True, help="Path to the first image.", default="/ghome/c5mcv01/mcv-c6-2025-team1/data/data_stereo_flow/training/image_0/000045_10.png")
    parser.add_argument('-im2', '--img2_path', type=str, required=True, help="Path to the second image.", default="/ghome/c5mcv01/mcv-c6-2025-team1/data/data_stereo_flow/training/image_0/000045_11.png")
    args = parser.parse_args()

    # Run the main function
    main(args)
