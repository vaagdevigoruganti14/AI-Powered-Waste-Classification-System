#!/usr/bin/env python3
"""
Simplified Smart Waste Classifier App
This version loads the model more efficiently
"""

import io
import json
import os
from typing import Dict

import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
MODEL_PATH = "waste_classifier_cnn.h5"
CLASS_INDICES_PATH = "class_indices.json"
IMAGE_SIZE = (128, 128)

# Global variables
model = None
idx_to_class = None

def load_class_mapping() -> Dict[int, str]:
    """Load class mapping from JSON file"""
    if os.path.exists(CLASS_INDICES_PATH):
        with open(CLASS_INDICES_PATH, "r", encoding="utf-8") as f:
            cls_to_idx = json.load(f)
        idx_to_cls = {int(v): k for k, v in cls_to_idx.items()}
        return idx_to_cls
    # Default fallback
    return {0: "hazardous", 1: "organic", 2: "recyclable"}

def load_model():
    """Load the TensorFlow model"""
    global model, idx_to_class
    
    if model is None:
        print("🔄 Loading AI model...")
        try:
            import tensorflow as tf
            if not os.path.exists(MODEL_PATH):
                raise RuntimeError("Model file not found. Please train the model first.")
            
            model = tf.keras.models.load_model(MODEL_PATH)
            idx_to_class = load_class_mapping()
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise

def preprocess(img: Image.Image) -> np.ndarray:
    """Preprocess image for model input"""
    img = img.convert("RGB").resize(IMAGE_SIZE)
    arr = np.asarray(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

@app.route("/", methods=["GET"]) 
def index():
    """Serve the main page"""
    frontend_path = os.path.join(os.path.dirname(__file__), "index.html")
    if not os.path.exists(frontend_path):
        fallback_path = os.path.join(os.path.dirname(__file__), "frontend.html")
        if not os.path.exists(fallback_path):
            return jsonify({"error": "Frontend not found"}), 404
        return send_file(fallback_path)
    return send_file(frontend_path)

@app.route("/health", methods=["GET"]) 
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "ok", 
        "message": "Smart Waste Classifier API is running",
        "model_loaded": model is not None
    }), 200

@app.route("/static/<path:filename>")
def static_files(filename):
    """Serve static files"""
    return send_file(os.path.join("static", filename))

@app.route("/api/classes", methods=["GET"])
def get_classes():
    """Get available waste classification classes"""
    if idx_to_class is None:
        load_class_mapping()
    return jsonify({
        "classes": list(idx_to_class.values()) if idx_to_class else [],
        "class_mapping": idx_to_class or {}
    }), 200

@app.route("/predict", methods=["POST"]) 
def predict():
    """Classify waste image"""
    try:
        # Load model if not already loaded
        if model is None:
            load_model()
        
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        f = request.files["file"]
        if f.filename == "":
            return jsonify({"error": "Empty filename"}), 400
        
        # Process image
        img = Image.open(io.BytesIO(f.read()))
        x = preprocess(img)
        
        # Make prediction
        preds = model.predict(x, verbose=0)[0]
        idx = int(np.argmax(preds))
        confidence = float(preds[idx]) * 100.0
        label = idx_to_class.get(idx, str(idx))
        
        return jsonify({
            "label": label, 
            "confidence": f"{confidence:.2f}"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/userinfo", methods=["POST"]) 
def userinfo():
    """Save user information"""
    try:
        data = request.get_json(force=True)
        print("User info:", data)
        return jsonify({"ok": True}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    print("♻️  Smart Waste Classifier")
    print("=" * 40)
    print("🚀 Starting application...")
    print("📍 Application will be available at: http://127.0.0.1:5000")
    print("🔄 Press Ctrl+C to stop the application")
    print("-" * 50)
    
    # Start the Flask app
    app.run(host="0.0.0.0", port=5000, debug=True)
