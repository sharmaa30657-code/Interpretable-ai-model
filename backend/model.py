import os
import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

CLASSES_PATHS = [
    os.path.join(BASE_DIR, "classes.pkl"),
    os.path.join(ROOT_DIR, "classes.pkl")
]

MODEL_CANDIDATES = [
    os.path.join(BASE_DIR, "best_model.pth"),
    os.path.join(BASE_DIR, "custom_model.pth"),
    os.path.join(ROOT_DIR, "best_model.pth"),
    os.path.join(ROOT_DIR, "custom_model.pth")
]

classes = None
for path in CLASSES_PATHS:
    if os.path.exists(path):
        with open(path, "rb") as f:
            classes = pickle.load(f)
        print(f"Loaded class labels from {path}")
        break

if classes is None:
    classes = ['audi', 'book', 'cat', 'Headphone', 'laptop', 'table', 'tiger', 'water bottle']
    print("Warning: classes.pkl not found. Using fallback labels.")

num_classes = len(classes)

MODEL_PATH = next((p for p in MODEL_CANDIDATES if os.path.exists(p)), None)

if MODEL_PATH is None:
    raise FileNotFoundError(
        "No model checkpoint found. Expected one of: " + ", ".join(MODEL_CANDIDATES)
    )

print(f"Loading model weights from {MODEL_PATH}")
checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))

if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
else:
    state_dict = checkpoint

if 'fc.weight' in state_dict:
    num_classes = state_dict['fc.weight'].shape[0]
elif 'fc.bias' in state_dict:
    num_classes = state_dict['fc.bias'].shape[0]
else:
    num_classes = len(classes)

loaded_classes = None
for path in CLASSES_PATHS:
    if os.path.exists(path):
        with open(path, 'rb') as f:
            candidate = pickle.load(f)
        if len(candidate) == num_classes:
            loaded_classes = candidate
            classes = candidate
            print(f"Loaded class labels from {path}")
            break
        else:
            print(f"Skipping {path}: {len(candidate)} labels does not match checkpoint output size {num_classes}")

if loaded_classes is None:
    if classes is not None and len(classes) == num_classes:
        loaded_classes = classes
    else:
        classes = [f"class_{i}" for i in range(num_classes)]
        print(f"Using generated class labels for {num_classes} classes")

model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(state_dict)
model.to(torch.device('cpu'))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
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

