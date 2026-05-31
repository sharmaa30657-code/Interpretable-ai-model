import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from flask import Flask, request, jsonify
from .model import predict_image, model

app = Flask(__name__, static_folder="static", template_folder="templates")

ENABLE_GRADCAM = os.environ.get("ENABLE_GRADCAM", "true").lower() in ("1", "true", "yes")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html")
@app.route("/upload", methods=["POST"])
def upload_image():

    if "image" not in request.files:
        return render_template("index.html")

    file = request.files["image"]

    if file.filename == "":
        return render_template("index.html")

    filepath = os.path.join("uploads", file.filename)

    file.save(filepath)

    prediction, confidence = predict_image(filepath)
    heatmap = None

    if ENABLE_GRADCAM:
        try:
            from .gradcam_utils import generate_gradcam

            heatmap = generate_gradcam(model, filepath)
        except Exception as e:
            heatmap = None
            app.logger.error(f"Grad-CAM failed: {e}")

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=round(confidence * 100, 2),
        heatmap=heatmap
    )


@app.route("/health")
def health():
    return "OK", 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)