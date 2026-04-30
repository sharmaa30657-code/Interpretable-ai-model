import os

if __package__ in (None, ""):
    from model import predict_image, model
else:
    from .model import predict_image, model

from flask import render_template
from flask import Flask, request, jsonify

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
        if __package__ in (None, ""):
            from gradcam_utils import generate_gradcam
        else:
            from .gradcam_utils import generate_gradcam

        heatmap = generate_gradcam(model, filepath)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=round(confidence * 100, 2),
        heatmap=heatmap
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)