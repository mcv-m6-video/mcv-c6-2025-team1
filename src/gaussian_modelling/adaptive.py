import cv2
import numpy as np

from gaussian_modelling.base import GaussianModelling


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
        # Convert to grayscale for processing
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Update the mean and variance using the rho parameter for the bg pixels
        self.mean[~mask] = self.rho * gray_img[~mask] + (1 - self.rho) * self.mean[~mask]
        self.variance[~mask] = self.rho * (gray_img[~mask] - self.mean[~mask]) ** 2 + (1 - self.rho) * self.variance[~mask]
    
        