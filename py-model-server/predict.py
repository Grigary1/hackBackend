# import torch
# from torchvision import transforms
# from PIL import Image
# from model import WasteClassifier

# # Step 1: Load the model
# model = WasteClassifier(num_classes=5)  # same as what you used while training

# # Step 2: Load weights correctly (trained as resnet18, so load into model.model)
# state_dict = torch.load("model.pth", map_location="cpu")
# model.model.load_state_dict(state_dict, strict=False)

# model.eval()

# # Step 3: Define transforms
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Standard for ResNet
#                          std=[0.229, 0.224, 0.225])
# ])

# # Step 4: Load and preprocess image
# img = Image.open("waste.jpeg").convert("RGB")
# input_tensor = transform(img).unsqueeze(0)  # shape: (1, 3, 224, 224)

# # Step 5: Predict
# with torch.no_grad():
#     output = model(input_tensor)
#     predicted_class = output.argmax(dim=1).item()

# print(f"Predicted class: {predicted_class}")



# predict.py
import torch
from torchvision import transforms
from PIL import Image
from model import WasteClassifier
from flask import Flask, request, jsonify
import io

# Class names (adjust based on your labels)
class_names = ['Plastic', 'Glass', 'Metal', 'Paper', 'Other']

# Initialize Flask app
app = Flask(__name__)

# Load model
model = WasteClassifier(num_classes=5)
state_dict = torch.load("model.pth", map_location="cpu")
model.model.load_state_dict(state_dict, strict=False)
model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'success': False, 'message': 'No image provided'}), 400

    file = request.files['image']
    img_bytes = file.read()
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        pred_index = outputs.argmax(1).item()
        pred_label = class_names[pred_index]

    return jsonify({'success': True, 'predicted_class': pred_label, 'class_index': pred_index})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
