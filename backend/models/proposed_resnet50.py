import torch
import torch.nn as nn
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ProposedResNet(nn.Module):
    def __init__(self, base_model, num_classes=18, dropout_rate=0.5):
        super(ProposedResNet, self).__init__()
        self.base_model = nn.Sequential(*list(base_model.children())[:-2])  # Exclude top layers
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.gmp = nn.AdaptiveMaxPool2d((1, 1))
        self.bn = nn.BatchNorm1d(4096)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(4096, num_classes)  # 2048 (gap) + 2048 (gmp) = 4096

    def forward(self, x):
        x = self.base_model(x)

        gap = self.gap(x).view(x.size(0), -1)
        gmp = self.gmp(x).view(x.size(0), -1)
        concat = torch.cat((gap, gmp), dim=1)
        
        bn = self.bn(concat)
        dp = self.dropout(bn)
        
        x = self.fc(dp)

        return x

class PROPOSED_RESNET50:
    def __init__(self, model_weights_path="backend/models/weights/torch_weights_baselineb32auglr001notenhancedvallacc98testacc98traintime4m_25s.pth"):
        self.torch_baseline = models.resnet50(weights="IMAGENET1K_V1")
        self.model = ProposedResNet(self.torch_baseline).to(device)
        self.model.load_state_dict(torch.load(model_weights_path))
        self.model.eval()

    def predict(self, image):
        with torch.no_grad():
            output = self.model(image)
            predicted_class = output.argmax(dim=1).item()
            return predicted_class
