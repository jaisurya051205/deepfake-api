import os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
from tensorflow.keras.models import load_model

# -----------------------------
# App Setup
# -----------------------------
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -----------------------------
# Load Model (only once)
# -----------------------------
MODEL_PATH = "deepfake_model.h5"   # change if your model name is different
model = load_model(MODEL_PATH)

# -----------------------------
# Frame Extraction Function
# -----------------------------
def extract_frames(video_path, max_frames=20):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (224, 224))
        frame = frame / 255.0
        frames.append(frame)

    cap.release()

    return np.array(frames)


# -----------------------------
# Health Check Route
# -----------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Deepfake API is running 🚀"})


# -----------------------------
# Predict Route
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files["video"]

    video_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(video_path)

    frames = extract_frames(video_path)

    if len(frames) == 0:
        return jsonify({"error": "No frames extracted"}), 400

    try:
        predictions = model.predict(frames)
        confidence = float(np.mean(predictions))

        label = "FAKE" if confidence > 0.7 else "REAL"

        return jsonify({
            "prediction": label,
            "confidence": round(confidence, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -----------------------------
# Run App (IMPORTANT FOR RENDER)
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)