import os

if __package__ in (None, ""):

    from gradcam_utils import generate_gradcam
    from model import predict_image, model
else:
    from .gradcam_utils import generate_gradcam
    from .model import predict_image, model

from flask import render_template
from flask import Flask, request, jsonify

app = Flask(__name__, static_folder="static", template_folder="templates")

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
    heatmap = generate_gradcam(model, filepath)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=round(confidence * 100, 2),
        heatmap=heatmap
    )

if __name__ == "__main__":
    app.run(debug=True)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)