import time
import torch
import torch.nn as nn
from torchvision.models import vgg16
from backend.utils.helpers import preprocess_image_for_classifier

class VGG16:
    def __init__(self, model_path="backend/models/weights/vgg16_full_model_5epoch.pth", device='cpu'):
        self.device = torch.device(device)

        self.model = torch.load(model_path, weights_only=False, map_location=device)
        # self.model.classifier[6] = nn.Linear(4096, 18)  # Modify classifier

        # self.model.load_state_dict(torch.load(model_path, map_location=self.device))

        self.model = self.model.to(self.device)
        self.model.eval()

    def predict(self, image):
        """Runs inference on the given image and returns class index, confidence, and processing time."""
        image = preprocess_image_for_classifier(image).to(self.device)

        start_time = time.time()
        
        with torch.no_grad():
            output = self.model(image)

        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted_class_index = probabilities.argmax(dim=1).item()
        confidence = probabilities.max().item()

        prediction_time = time.time() - start_time

        return predicted_class_index, confidence, prediction_time, probabilities[0].cpu().numpy()
