import gradio as gr

from backend.model import predict_image, model
from backend.gradcam_utils import generate_gradcam

def predict(img):

    prediction, confidence = predict_image(img)

    heatmap = generate_gradcam(model, img)

    result = f"Prediction: {prediction}\nConfidence: {confidence*100:.2f}%"

    return result, heatmap

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="filepath"),
    outputs=[
        gr.Textbox(label="Prediction Result"),
        gr.Image(label="Grad-CAM Visualization")
    ],
    title="Interpretable AI Model",
    description="Upload an image to see predictions with Grad-CAM visualization"
)

demo.launch()