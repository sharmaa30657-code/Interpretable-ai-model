import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn

# classes
classes = ['audi', 'cat', 'tiger']

# model load
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 3)

model.load_state_dict(torch.load("custom_model.pth", map_location=torch.device('cpu')))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])
def predict_image(img_path):

    image = Image.open(img_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)

        confidence, predicted = torch.max(probs, 0)

    return classes[predicted.item()], confidence.item()

