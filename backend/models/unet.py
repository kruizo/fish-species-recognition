import numpy as np
import torch
from backend.utils.helpers import preprocess_image_for_segment
import tensorflow as tf


class UNET:
    def __init__(self, model_path="backend/models/weights/unet_model224.pth", device='cpu'):
        self.model = torch.load(model_path, weights_only=False)
        self.device = torch.device(device)
        self.model.to(device)

    def predict(self, image):
        self.model.eval()
        image = preprocess_image_for_segment(image).to(self.device)

        with torch.no_grad():
            self.model.eval()  
            predicted_mask = self.model(image)

        # Convert the tensor to a NumPy array and remove the batch dimension (assuming batch_size=1)
        predicted_mask = predicted_mask.squeeze().cpu().numpy()

        # Ensure the output has a single channel (1, 224, 224)
        binary_mask = (predicted_mask > 3).astype(np.uint8)



        return binary_mask


