import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from torchvision.models import resnext101_32x8d


class FeatureExtractor:
    """
    Class to extract features from a frame using a pre-trained model
    """
    def __init__(self, model_path = "/ghome/c5mcv01/mcv-c6-2025-team1/week4/src/algoritm_features/model_reid/resnext101_ibn_a_2.pth"):
        """
        Initializes the feature extractor
        Args:
            model_path: Path to the trained model (.pth)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Carreguem el model base
        self.model = resnext101_32x8d(weights=None)  # No cal que sigui preentrenat

        # Carreguem els pesos des del fitxer .pth
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict, strict=False)  # strict=False per si falten algunes capes

        # Eliminem l'última capa fully connected
        self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))
        self.model = self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extract_features(self, frame: Image) -> np.ndarray:
        """
        Extract features from a frame using the model
        Args:
            frame: Frame to extract features from

        Returns: Feature vector of the frame
        """
        input_tensor = self.transform(frame).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)

        feature_vector = output.squeeze().cpu().numpy()
        return feature_vector