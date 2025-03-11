import numpy as np
import cv2
import time
import argparse
import pyflow
import ptlflow
import torch.nn as nn

from src.optical_flow.plots import hsv_plot, visualize_arrow
from src.optical_flow.utils import calculate_msen, calculate_pepn, read_png_file
from ptlflow.utils.io_adapter import IOAdapter


class OpticalFlow:
    def __init__(self, model: str, params: list = None):
        """Optical flow processor class.

        Args:
            model (str): Model name to use for computing the optical flow.
            params (list, optional): Parameters only for pyflow. Defaults to None.
        """
        self.supported_models = ["pyflow", "dicl", "memflow", "rapidflow", "rpknet", "dip"]
        assert model in self.supported_models, "This model is not supported, please choose another one..."
        
        self.model_name = model
        self.is_pyflow = True if model == "pyflow" else False
        
        self.model = None
        if not self.is_pyflow:
            self.model = self.__get_model(model)
            
        # Params for pyflow
        self.params = params
        
    def __get_model(self, model: str) -> ptlflow.BaseModel:
        """Returns the model class for PLTFlow

        Args:
            model (str): Model name

        Returns:
            pltflow.BaseModel: BaseModel class from PLTFlow
        """
        model_ptlflow = ptlflow.get_model(model, ckpt_path='things')
        return model_ptlflow
    
    def compute_flow(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """Computes the optical flow based on the chosen model and parameters

        Args:
            img1 (np.ndarray): First image array
            img2 (np.ndarray): Second image array

        Returns:
            np.ndarray: Stack of horizontal and vertical flow.
        """
        if self.is_pyflow:
            alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations, nSORIterations, colType = self.params
            u, v, _ = pyflow.coarse2fine_flow(
                img1, img2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations, nSORIterations, colType
            )
        else:
            # Preprocess images 
            img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)  # expects RGB images
            img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            
            #self.model = nn.DataParallel(self.model) # To check if it works for multi-GPU support
            self.model.to("cuda")
            fp16 = False
            if self.model_name != "dicl":
                self.model = self.model.half()
                fp16 = True
            io_adapter = IOAdapter(self.model, img1_rgb.shape[:2], cuda=True, fp16=fp16)
            inputs = io_adapter.prepare_inputs([img1_rgb, img2_rgb])
            
            # Forward the inputs through the model
            predictions = self.model(inputs)
            
            flow = predictions['flows'][0, 0]  # Remove batch and sequence dimensions
            flow = flow.permute(1, 2, 0)  # Change from CHW to HWC shape
            
            u = flow[:, :, 0].detach().cpu().numpy()  # Extract the horizontal flow component (u)
            v = flow[:, :, 1].detach().cpu().numpy()  # Extract the vertical flow component (v)
        return np.dstack((u, v))
    
    def run_model(self, im1_path: str, img1: np.ndarray, img2: np.ndarray, gt: np.ndarray) -> tuple[float, float, float]:
        """Runs the selected optical flow model, computes MSEN, PEPN and inference time

        Args:
            im1_path (str): Image file path
            img1 (np.ndarray): First image array
            img2 (np.ndarray): Second image array
            gt (np.ndarray): Ground Truth array.

        Returns:
            tuple: Tuple of metrics (MSEN, PEPN) and inference time.
        """
        # Compute flow
        start_time = time.time()
        flow = self.compute_flow(img1, img2)
        inference_time = time.time() - start_time
        
        # Calculate MSEN and PEPN
        msen = calculate_msen(gt, flow)
        pepn = calculate_pepn(gt, flow)

        # Visualize and save the results
        hsv_plot(flow, filename=self.model)
        hsv_plot(gt, filename="GT")
        
        visualize_arrow(im1_path, gt, filename="GT")
        visualize_arrow(im1_path, flow, filename=self.model_name)
        
        return msen, pepn, inference_time


def main(args):
    # Fixed image paths and ground truth flow path
    gt_path = args.gt_path
    im1_path = args.img1_path
    im2_path = args.img2_path
    gt = read_png_file(gt_path)

    # Define model parameters (example for PyFlow, modify as needed)
    if args.model == "pyflow":
        img1 = cv2.imread(im1_path, cv2.IMREAD_GRAYSCALE)  
        img2 = cv2.imread(im2_path, cv2.IMREAD_GRAYSCALE)
        img1 = np.atleast_3d(img1.astype(float) / 255.0)
        img2 = np.atleast_3d(img2.astype(float) / 255.0)
        params = [0.012, 0.75, 20, 1, 1, 30, 1]  # Example: alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations, nSORIterations, colType
    elif args.model == "dicl" or args.model == "memflow" or args.model == "rapidflow" or args.model =="rpknet" or args.model == "dip":
        img1 = cv2.imread(im1_path, cv2.IMREAD_GRAYSCALE)  
        img2 = cv2.imread(im2_path, cv2.IMREAD_GRAYSCALE)
        params = []
    else:
        print(f"Model {args.model} not implemented yet.")
        return

    # Run the model and compute metrics
    optical_flow = OpticalFlow(args.model, params)
    msen, pepn, inference_time = optical_flow.run_model(im1_path, img1, img2, gt)

    # Print results
    print(f"\nModel: {args.model}")
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
