import torch.nn as nn
from torchvision import models

class AudioResNet(nn.Module):
    def __init__(self, architecture='resnet18', num_classes=2):
        super(AudioResNet, self).__init__()
        
        # 1. Initialize the base model without pretrained weights
        if architecture == 'resnet18':
            self.model = models.resnet18(weights=None)
        elif architecture == 'resnet34':
            self.model = models.resnet34(weights=None)
        elif architecture == 'resnet50':
            self.model = models.resnet50(weights=None)
        else:
            raise ValueError("Architecture not supported. Choose resnet18, resnet34, or resnet50.")
            
        # 2. Adjust the first convolution layer for 1 input channel
        # This is standard across all ResNet variants to handle grayscale signal maps.
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # 3. Modify the Fully Connected layer for the correct number of classes
        # This dynamically finds 'in_features' (512 for R18/34, 2048 for R50).
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
            
    def forward(self, x):
        return self.model(x)