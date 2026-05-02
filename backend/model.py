import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn
import pickle

# Load classes
try:
    with open("classes.pkl", "rb") as f:
        classes = pickle.load(f)
except FileNotFoundError:
    classes = ['audi', 'book', 'cat', 'Headphone', 'laptop', 'table', 'tiger', 'water bottle']  # Fallback

num_classes = len(classes)

# model load
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, num_classes)

model.load_state_dict(torch.load("custom_model.pth", map_location=torch.device('cpu')))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
def predict_image(img_path):

    image = Image.open(img_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)

        confidence, predicted = torch.max(probs, 0)

    return classes[predicted.item()], confidence.item()

