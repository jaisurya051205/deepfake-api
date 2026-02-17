from flask_cors import CORS
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import os
import cv2

# -----------------------------
# Flask App
# -----------------------------
app = Flask(__name__)
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "backend/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_PATH = "ml_model/deepfake_model.h5"

# -----------------------------
# Load Model (once)
# -----------------------------
model = load_model(MODEL_PATH)

# -----------------------------
# Home Route (Fixes 404)
# -----------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "OK",
        "message": "Deepfake Detection API is running"
    })

# -----------------------------
# Helper: Extract frames
# -----------------------------
def extract_frames(video_path, max_frames=10):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0

    while cap.isOpened() and count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (128, 128))
        frame = frame / 255.0
        frames.append(frame)
        count += 1

    cap.release()
    return np.array(frames)

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

    predictions = model.predict(frames)
    confidence = float(np.mean(predictions))
    print("Mean confidence:", confidence)

    label = "FAKE" if confidence > 0.5 else "REAL"

    return jsonify({
        "prediction": label,
        "confidence": round(confidence, 2)
    })

# -----------------------------
# Run Server
# -----------------------------
if __name__ == "__main__":
   app.run(host="0.0.0.0", port=5000, debug=True)