import cv2
import torch
import time
import torchvision.transforms as transforms
from torchvision import models
from torch import nn
import os

# Define the 23 material class labels
class_labels = [
    'aluminum_foil', 'asphalt', 'brick', 'cardboard', 'carpet', 'ceramic',
    'concrete', 'fabric', 'foliage', 'food', 'glass', 'grass', 'gravel',
    'hair', 'leather', 'metal', 'mirror', 'paper', 'plastic', 'polished_wood',
    'soil', 'stone', 'wood'
]

# Image transform
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model architecture
model = models.resnet18()
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(class_labels))

# Load trained weights
model_path = os.path.join(os.path.dirname(__file__), "material_classifier_resnet18.pt")
model.load_state_dict(torch.load(model_path, map_location=device))

# Send model to device
model = model.to(device)
model.eval()

# Start webcam
cap = cv2.VideoCapture(0)

print("Real-time Material Detection Started. Press ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame
    image = transform(frame).unsqueeze(0).to(device)

    # Inference
    start_time = time.time()
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    end_time = time.time()

    material = class_labels[predicted.item()]
    fps = 1 / (end_time - start_time)

    # Annotate frame
    cv2.putText(frame, f"Material: {material}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 255, 50), 2)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Show result
    cv2.imshow('Material Detector', frame)

    # Press ESC to quit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
