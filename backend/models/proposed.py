import time
import torch
import torch.nn as nn
from torchvision.models import resnet50
from backend.utils.helpers import preprocess_image_for_classifier


class ProposedResNet(nn.Module):
    def __init__(self, base_model, num_classes=18, dropout_rate=0.25):
        super(ProposedResNet, self).__init__()
        self.base_model = nn.Sequential(*list(base_model.children())[:-2])  # Exclude top layers
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.gmp = nn.AdaptiveMaxPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(4096, num_classes)  # 2048 (gap) + 2048 (gmp) = 4096

    def forward(self, x):
        x = self.base_model(x)
        gap = self.gap(x).view(x.size(0), -1)
        gmp = self.gmp(x).view(x.size(0), -1)
        concat = torch.cat((gap, gmp), dim=1)
        
        x = self.fc(concat)

        return x
    
class PROPOSED_RESNET50:
    def __init__(self, model_path="backend/models/weights/FINAL_torch_weights_proposedb16notauglr001ENHANCEDwUNETtraintime_freezeAllEpoch30withclassweights.pth", device='cpu'):
        # self.model = torch.jit.load(model_path, map_location=device)
        self.model = resnet50(weights="IMAGENET1K_V1")
        self.device = torch.device(device)
        self.model = ProposedResNet(self.model).to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

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
        

