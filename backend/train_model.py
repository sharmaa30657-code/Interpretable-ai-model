import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import pickle

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder("../dataset", transform=transform)

classes = dataset.classes
num_classes = len(classes)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_data, test_data = torch.utils.data.random_split(dataset,[train_size,test_size])

train_loader = DataLoader(train_data,batch_size=8,shuffle=True)
test_loader = DataLoader(test_data,batch_size=8)

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.0001)

best_accuracy = 0.0
best_model_path = os.path.join(os.path.dirname(__file__), "best_model.pth")

for epoch in range(20):
    running_loss = 0
    model.train()
    for images,labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1} Loss: {running_loss:.4f}")

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Validation Accuracy: {accuracy:.2f}%")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), best_model_path)
        print(f"Saved best model with accuracy {accuracy:.2f}%")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "custom_model.pth")
torch.save(model.state_dict(), MODEL_PATH)

CLASSES_PATH = os.path.join(os.path.dirname(__file__), "classes.pkl")
with open(CLASSES_PATH, "wb") as f:
    pickle.dump(classes, f)

print(f"Model training complete! Saved weights to {MODEL_PATH} and classes to {CLASSES_PATH}")
print(f"Best validation accuracy: {best_accuracy:.2f}%")