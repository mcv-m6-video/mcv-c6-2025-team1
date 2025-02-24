import cv2
import numpy as np

from gaussian_modelling.base import GaussianModelling
import time
import jax


class AdaptiveGaussianModelling(GaussianModelling):
    def __init__(self, alpha: float=0.01, rho: float=0.1, use_median: bool=False):
        """Initializes the Adaptive Gaussian Modelling algorithm

        Args:
            alpha (float): Alpha parameter for the Gaussian Modelling algorithm
            rho (float): Learning rate for the adaptive model
            use_median (bool): Whether to use the median instead of the mean. Defaults to False.
        """
        super().__init__(alpha, use_median)
        self.rho = rho
    
    def update_model(self, frame: np.ndarray, mask: np.ndarray):
        """Update the background model based on the new frame

        Args:
            frame (np.ndarray): The current frame to update the model with
            mask (np.ndarray): The mask defining bg and fg pixels
        """
        #start_time = time.time()  # Record start time
        # Convert to grayscale for processing
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Update the mean and variance using the rho parameter for the bg pixels
        mean = self.rho * gray_img + (1 - self.rho) * self.mean
        var = self.rho * (gray_img - self.mean) ** 2 + (1 - self.rho) * self.variance
        bg_mask = (mask == 0)

        if isinstance(self.mean, jax.Array):
            self.mean = self.mean.at[bg_mask].set(mean[bg_mask])
            self.variance = self.variance.at[bg_mask].set(var[bg_mask])
        elif isinstance(self.mean, np.ndarray):
            self.mean[bg_mask] = mean[bg_mask]
            self.variance[bg_mask] = var[bg_mask]
            
            
        #end_time = time.time()  # Record end time
        #elapsed_time = end_time - start_time
        #print(f"Execution time: {elapsed_time:.4f} seconds")
    
        