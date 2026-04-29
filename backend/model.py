import os
import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn
import pickle


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

CLASSES_PATHS = [
    os.path.join(ROOT_DIR, "classes.pkl"),
    os.path.join(BASE_DIR, "classes.pkl")
]


MODEL_CANDIDATES = [
    os.path.join(ROOT_DIR, "best_model.pth"),
    os.path.join(BASE_DIR, "best_model.pth"),
    os.path.join(ROOT_DIR, "custom_model.pth"),
    os.path.join(BASE_DIR, "custom_model.pth")
]


classes = None

for path in CLASSES_PATHS:
    if os.path.exists(path):
        with open(path, "rb") as f:
            classes = pickle.load(f)
        print(f"Loaded class labels from {path}")
        break


MODEL_PATH = next(
    (p for p in MODEL_CANDIDATES if os.path.exists(p)),
    None
)


num_classes = len(classes) if classes is not None else 0


if num_classes == 0 and MODEL_PATH is not None:

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=torch.device("cpu")
    )

    if isinstance(checkpoint, dict):

        if "fc.weight" in checkpoint:
            num_classes = checkpoint["fc.weight"].shape[0]

        elif (
            "state_dict" in checkpoint and
            "fc.weight" in checkpoint["state_dict"]
        ):
            num_classes = checkpoint["state_dict"]["fc.weight"].shape[0]

if num_classes == 0:
    num_classes = 3
    classes = [f"class_{i}" for i in range(num_classes)]

    print(
        "Warning: Could not load class labels or checkpoint info. "
        "Falling back to 3 classes."
    )

elif classes is None:
    classes = [f"class_{i}" for i in range(num_classes)]


model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    num_classes
)


if MODEL_PATH and os.path.exists(MODEL_PATH):

    print(f"Loading model weights from {MODEL_PATH}")

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=torch.device("cpu")
        )
    )

    model.to(torch.device("cpu"))
    model.eval()

else:

    print(
        "Warning: Model weights not found in backend or project root. "
        "Using randomly initialized model."
    )

    model.to(torch.device("cpu"))
    model.eval()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def predict_image(img_path):

    image = Image.open(img_path).convert("RGB")

    image = transform(image).unsqueeze(0)

    with torch.no_grad():

        outputs = model(image)

        probs = torch.nn.functional.softmax(
            outputs[0],
            dim=0
        )

        confidence, predicted = torch.max(probs, 0)

    predicted_class = classes[predicted.item()]

    return predicted_class, confidence.item()