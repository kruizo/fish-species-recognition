import time
import torch
import torch.nn as nn
from torchvision import models
from backend.utils.helpers import preprocess_image_for_classifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BaselineResNet(nn.Module):
    def __init__(self, base_model, num_classes=18):
        super(BaselineResNet, self).__init__()
        self.base_model = nn.Sequential(*list(base_model.children())[:-2])  
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)  

    def forward(self, x):
        x = self.base_model(x)
        x = self.gap(x).view(x.size(0), -1)  
        x = self.fc(x) 
        return x

class BASELINE_RESNET50:
    def __init__(self, model_path="backend/models/weights/torch_baselineb32notauglr001notenhancedvallacc82testacc80traintime3m_2sepoch5ratio70_20_10epoch5.pth", device=device):
        self.model = torch.load(model_path, weights_only=False)
        self.device = torch.device(device)
        self.model.to(device)

    def predict(self, image):
        image = preprocess_image_for_classifier(image)

        start_time = time.time()
        
        with torch.no_grad():
            self.model.eval()
            output = self.model(image)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            predicted_class_index = probabilities.argmax(dim=1).item()
            confidence = probabilities.max().item()
            
        prediction_time = time.time() - start_time

        return predicted_class_index, confidence, prediction_time



