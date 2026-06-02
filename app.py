from flask import Flask, render_template, request
from predict import predict_gender
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return "No file uploaded"

    file = request.files["image"]

    if file.filename == "":
        return "No file selected"

    filename = secure_filename(file.filename)

    filepath = os.path.join(UPLOAD_FOLDER, filename)

    file.save(filepath)

    print("Saved:", filepath)
    print("Exists:", os.path.exists(filepath))

    gender, confidence = predict_gender(filepath)

    return render_template(
        "index.html",
        prediction=f"{gender} ({confidence:.2f}%)",
        image_path="/" + filepath.replace("\\", "/")
    )

if __name__ == "__main__":
    app.run(debug=True)