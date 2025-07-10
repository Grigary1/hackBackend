# You used ResNet18 with a custom final layer
import torch.nn as nn
import torchvision.models as models

class WasteClassifier(nn.Module):
    def __init__(self, num_classes=5):  # change to your actual class count
        super(WasteClassifier, self).__init__()
        self.model = models.resnet18(weights='IMAGENET1K_V1')  # pretrained=True equivalent
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
