import io
import json
import os
from typing import Dict

import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import tensorflow as tf


MODEL_PATH = os.path.join("waste_classifier_cnn.h5")
CLASS_INDICES_PATH = os.path.join("class_indices.json")
IMAGE_SIZE = (128, 128)


def load_class_mapping() -> Dict[int, str]:
    if os.path.exists(CLASS_INDICES_PATH):
        with open(CLASS_INDICES_PATH, "r", encoding="utf-8") as f:
            cls_to_idx = json.load(f)
        idx_to_cls = {int(v): k for k, v in cls_to_idx.items()}
        return idx_to_cls
    # default order fallback
    return {0: "hazardous", 1: "organic", 2: "recyclable"}


app = Flask(__name__)
CORS(app)

model = None
idx_to_class = load_class_mapping()


def load_model_once() -> None:
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(
                "Model file not found. Train the model first by running 'python train.py'"
            )
        model = tf.keras.models.load_model(MODEL_PATH)


def preprocess(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize(IMAGE_SIZE)
    arr = np.asarray(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


@app.route("/health", methods=["GET"]) 
def health() -> tuple:
    return jsonify({"status": "ok"}), 200


@app.route("/", methods=["GET"]) 
def index() -> tuple:
    # Serve the enhanced frontend from project root
    frontend_path = os.path.join(os.path.dirname(__file__), "index.html")
    if not os.path.exists(frontend_path):
        # Fallback to original frontend.html
        fallback_path = os.path.join(os.path.dirname(__file__), "frontend.html")
        if not os.path.exists(fallback_path):
            return jsonify({"error": "Frontend not found"}), 404
        return send_file(fallback_path)
    return send_file(frontend_path)

@app.route("/static/<path:filename>")
def static_files(filename):
    """Serve static files like CSS, JS, images"""
    return send_file(os.path.join("static", filename))

@app.route("/api/classes", methods=["GET"])
def get_classes() -> tuple:
    """Get available waste classification classes"""
    return jsonify({
        "classes": list(idx_to_class.values()),
        "class_mapping": idx_to_class
    }), 200

@app.route("/predict", methods=["POST"]) 
def predict() -> tuple:
    load_model_once()
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    try:
        img = Image.open(io.BytesIO(f.read()))
        x = preprocess(img)
        preds = model.predict(x, verbose=0)[0]
        idx = int(np.argmax(preds))
        confidence = float(preds[idx]) * 100.0
        label = idx_to_class.get(idx, str(idx))
        return jsonify({"label": label, "confidence": f"{confidence:.2f}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/userinfo", methods=["POST"]) 
def userinfo() -> tuple:
    try:
        data = request.get_json(force=True)
        # In a real app, store this in DB; here we no-op/log.
        print("User info:", data)
        return jsonify({"ok": True}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)


