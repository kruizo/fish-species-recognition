import torch
import torch.nn as nn
from torchvision import models

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
    def __init__(self, model_weights_path="backend/models/weights/torch_weights_baselineb32auglr001notenhancedvallacc98testacc98traintime4m_25s.pth"):
        self.torch_baseline = models.resnet50(weights="IMAGENET1K_V1")
        self.model = BaselineResNet(self.torch_baseline).to(device)
        self.model.load_state_dict(torch.load(model_weights_path))
        self.model.eval()

    def predict(self, image):
        with torch.no_grad():
            output = self.model(image)
            predicted_class = output.argmax(dim=1).item()
            return predicted_class
