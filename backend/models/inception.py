import time
import torch
import torch.nn as nn
from torchvision.models import inception_v3
from backend.utils.helpers import preprocess_image_for_classifier

class INCEPTIONV3:
    def __init__(self, model_path="backend/models/weights/inception_model_30epoch.pth", device='cpu'):
        self.device = torch.device(device)

        self.model = torch.load(model_path, weights_only=False, map_location=device)
        # self.model.fc = nn.Linear(self.model.fc.in_features, 18) 
        
        # if self.model.aux_logits:
            # self.model.AuxLogits.fc = nn.Linear(self.model.AuxLogits.fc.in_features, 18)  
        # Load saved weights
        # self.model.load_state_dict(torch.load(model_path, map_location=self.device))

        # Move model to device
        self.model = self.model.to(self.device)
        self.model.eval()

    def predict(self, image):
        """Runs inference on the given image and returns class index, confidence, and processing time."""
        image = preprocess_image_for_classifier(image, (299, 299)).to(self.device)

        start_time = time.time()
        
        with torch.no_grad():
            output = self.model(image)

        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted_class_index = probabilities.argmax(dim=1).item()
        confidence = probabilities.max().item()

        prediction_time = time.time() - start_time

        return predicted_class_index, confidence, prediction_time, probabilities[0].cpu().numpy()
