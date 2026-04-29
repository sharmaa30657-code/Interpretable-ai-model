try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
except ImportError:
    GradCAM = None
    show_cam_on_image = None

import cv2
import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def generate_gradcam(model, img_path):
    if GradCAM is None or show_cam_on_image is None:
        raise ImportError(
            "pytorch_grad_cam is required. Run: pip install grad-cam"
        )

    image = Image.open(img_path).convert("RGB")
    img_tensor = _transform(image).unsqueeze(0)

    target_layer = model.layer4[-1]

    cam = GradCAM(model=model, target_layers=[target_layer])

    grayscale_cam = cam(input_tensor=img_tensor)[0]

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0

    cam_h, cam_w = grayscale_cam.shape
    img_resized = cv2.resize(img, (cam_w, cam_h))

    visualization = show_cam_on_image(img_resized, grayscale_cam, use_rgb=True)


    output_dir = os.path.join(os.path.dirname(__file__), "static")
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "gradcam_result.jpg")
    cv2.imwrite(output_path, visualization)

    del cam
    del img_tensor

    return "/static/gradcam_result.jpg"
