import torch.nn as nn
import torchvision.models as models

class WasteClassifier(nn.Module):
    def __init__(self, num_classes=2):  # change 2 to the actual number of your classes
        super(WasteClassifier, self).__init__()
        self.model = models.resnet18(weights=None)
        
        # Modify the final FC layer if necessary
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)
