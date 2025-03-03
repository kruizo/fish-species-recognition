import time
import torch
import torch.nn as nn
from torchvision.models import densenet121
from backend.utils.helpers import preprocess_image_for_classifier


class DENSENET121:
    def __init__(self, model_path="backend/models/weights/densenet_full_model_30epoch.pth", device='cpu'):
        self.model = torch.load(model_path, weights_only=False, map_location=device)
        # num_features = self.model.classifier.in_features
        # self.model.classifier = nn.Linear(num_features, 18)  # Same as your training script

        self.device = torch.device(device)
        self.model = self.model.to(device)
        # self.model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

    def predict(self, image):
        image = preprocess_image_for_classifier(image).to(self.device)

        start_time = time.time()

        with torch.no_grad():
            self.model.eval()
            output = self.model(image)

        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted_class_index = probabilities.argmax(dim=1).item()
        confidence = probabilities.max().item()

        prediction_time = time.time() - start_time

        return predicted_class_index, confidence, prediction_time, probabilities[0].cpu().numpy()
