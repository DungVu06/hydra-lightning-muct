import torch.nn as nn
from torchvision import models

class FaceLandmarkModel(nn.Module):
    def __init__(self, num_landmarks=76):
        super().__init__()
        # 1. Dùng một backbone mạnh mẽ có sẵn
        self.backbone = models.resnet18(pretrained=True)
        
        # 2. Thay đổi lớp FC cuối cùng để khớp với số điểm của MUCT
        # ResNet18 có 512 đặc trưng ở lớp cuối
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, num_landmarks * 2) 

    def forward(self, x):
        return self.backbone(x)