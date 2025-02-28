from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
from PIL import Image

import numpy as np 
import torch
import tensorflow as tf
import tensorflow_hub as hub

import cv2
import skimage

from matplotlib import pyplot as plt
import matplotlib.patches as patches



class SSD_VGG16:
    def __init__(self,  box_score_thresh: float=0.9):
        """SSD model using VGG-16 backbone.

        Args:
            box_score_thresh (float, optional): Threshold for box score. Defaults to 0.9.
        """
        self.model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd')
        self.utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')
        self.model.to('cuda')
        self.model.eval()
    
    def decode_img(self, image: np.ndarray) -> torch.Tensor:
        """Convert image from OpenCV to Tensor.

        Args:
            image (np.ndarray): Image in grayscale.

        Returns:
            torch.Tensor: Image in tensor format.
        """
        # TODO: Decode image from OpenCV to Tensor
        image = utils.prepare_tensor(image)
        return image

    def predict(self, image: np.ndarray) -> torch.Tensor:
        """Predict bounding boxes on an image.

        Args:
            image (np.ndarray): NumPy array of image in grayscale.

        Returns:
            torch.Tensor: Predictions from the model.
        """
        print("Predicting image with shape:", image.shape)
        
        # Convert image to tensor
        image = self.decode_img(image)
        
        with torch.no_grad():
            detections_batch = self.model(image)
            results_per_input = self.utils.decode_results(detections_batch)
            best_results_per_input = [utils.pick_best(results, 0.40) for results in results_per_input]
            classes_to_labels = self.utils.get_coco_object_dictionary()
        
        return best_results_per_input

    def draw_predictions(self, image: torch.Tensor | np.ndarray, prediction: torch.Tensor) -> Image:
        """Draw bounding boxes on an image.

        Args:
            image (torch.Tensor | np.ndarray): Image to draw bounding boxes on.
            prediction (torch.Tensor): Prediction for that frame from the model.

        Returns:
            Image: Image with bounding boxes drawn.
        """
        image = torch.from_numpy(image).permute(2, 0, 1)
        #image = image.unsqueeze(0)
        print("Image shape before drawing:", image.shape)
        
        labels = [self.weights.meta["categories"][i] for i in prediction["labels"]]
        
        image = draw_bounding_boxes(image, prediction['boxes'], 
                                    labels=labels, 
                                    colors="red", 
                                    width=4, font="DejaVuSans",
                                    font_size=30)
        image = to_pil_image(image.detach())
        return image



# Example usage
if __name__ == "__main__":
    ssd_model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd')
    utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')
    ssd_model.to('cuda')
    ssd_model.eval()
    # Let's load the video and get the first frame
    video = cv2.VideoCapture("/ghome/c3mcv02/mcv-c6-2025-team1/data/AICity_data/train/S03/c010/vdo.avi")
    ret, frame = video.read()
    
    # Predict bounding boxes on the first frame
    frame_resized = cv2.resize(frame, (300, 300))
    frame_rgb_1 = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)/255
    frame_rgb = np.expand_dims(frame_rgb_1, axis=0)
    tensor = utils.prepare_tensor(frame_rgb)
    with torch.no_grad():
        detections_batch = ssd_model(tensor)
        print(f"Detections_batch: {detections_batch}")
    
    results_per_input = utils.decode_results(detections_batch)
    print(f"Results: {results_per_input}")
    best_results_per_input = [utils.pick_best(results, 0.1) for results in results_per_input]
    classes_to_labels = utils.get_coco_object_dictionary()
    print(f"Classes to labels: {classes_to_labels}")

    # Create a new figure for each image
    fig, ax = plt.subplots(1)
        
    # Show the original, denormalized image
    image = frame_rgb_1
    ax.imshow(image)
        
    # Get bounding boxes, classes, and confidences
    bboxes, classes, confidences = best_results_per_input[0]
    print(f"bboxes: {bboxes}, classes: {classes}, confidences: {confidences}")   

    # Loop through the bounding boxes and draw them
    for idx in range(len(bboxes)):
        left, bot, right, top = bboxes[idx]
        x, y, w, h = [val * 300 for val in [left, bot, right - left, top - bot]]
        print(f"x{x}, y{y}, w{w}, h{h}")
            
        # Create a rectangle for the bounding box
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
            
        # Add text (class label and confidence) to the bounding box
        ax.text(x, y, "{} {:.0f}%".format(classes_to_labels[classes[idx] - 1], confidences[idx]*100), 
                bbox=dict(facecolor='white', alpha=0.5))
        
    # Save the image with detections as a new file
    output_image_path = f"image_with_predictions_{0}.png"
    plt.savefig(output_image_path)
    print(f"Image with predictions saved as {output_image_path}")
        
    # Close the figure to avoid memory issues if running in a loop
    plt.close(fig)

