import time
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2
from backend.utils.helpers import preprocess_image_for_classifier

class MOBILENET:
    def __init__(self, model_path="backend/models/weights/latestmobilenet_v2_model_30epoch.pth", device='cpu'):
        self.device = torch.device(device)

        # Load pre-trained MobileNetV2
        self.model = torch.load(model_path, weights_only=False, map_location=device)
        # Modify classifier for 18 classes
        # self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, 18)

        # Load trained weights
        # self.model.load_state_dict(torch.load(model_path, map_location=self.device), strict=False)
        self.model = self.model.to(self.device)
        self.model.eval()

    def predict(self, image):
        image = preprocess_image_for_classifier(image).to(self.device)

        start_time = time.time()

        with torch.no_grad():
            output = self.model(image)

        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted_class_index = probabilities.argmax(dim=1).item()
        confidence = probabilities.max().item()

        prediction_time = time.time() - start_time

        return predicted_class_index, confidence, prediction_time, probabilities[0].cpu().numpy()
