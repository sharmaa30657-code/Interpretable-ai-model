from flask import render_template
from model import predict_image
from flask import Flask, request, jsonify
import os

app = Flask(__name__)

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

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=round(confidence * 100, 2)
    )



    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    prediction, confidence = predict_image(filepath)

    return jsonify({
        "message": "Image uploaded successfully",
        "prediction": prediction,
        "confidence": round(confidence * 100, 2)
    })



    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    prediction = predict_image(filepath)

    return jsonify({
        "message": "Image uploaded successfully",
        "prediction_class_index": prediction
    })

if __name__ == "__main__":
    app.run(debug=True)