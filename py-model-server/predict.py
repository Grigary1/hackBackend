# predict.py
import sys
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

IMG_PATH = sys.argv[1]  # Get image path from Node.js
MODEL_PATH = os.path.join(os.path.dirname(__file__), "waste_model.pth")
NUM_CLASSES = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

image = Image.open(IMG_PATH).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(DEVICE)

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

with torch.no_grad():
    outputs = model(input_tensor)
    probs = torch.softmax(outputs, dim=1)
    pred_class = torch.argmax(probs, dim=1).item()

class_names = ['biodegradable', 'recyclable', 'hazardous', 'other']
print(json.dumps({
    "prediction": class_names[pred_class],
    "confidence": round(probs[0][pred_class].item(), 4)
}))
