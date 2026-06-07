#!/usr/bin/env python3
"""
Simple test version of the waste classifier app
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"]) 
def index():
    """Serve the main page"""
    frontend_path = os.path.join(os.path.dirname(__file__), "index.html")
    if not os.path.exists(frontend_path):
        return jsonify({"error": "Frontend not found"}), 404
    return send_file(frontend_path)

@app.route("/health", methods=["GET"]) 
def health():
    """Health check endpoint"""
    return jsonify({"status": "ok", "message": "Waste Classifier API is running"}), 200

@app.route("/static/<path:filename>")
def static_files(filename):
    """Serve static files"""
    return send_file(os.path.join("static", filename))

if __name__ == "__main__":
    print("🚀 Starting Smart Waste Classifier (Test Version)...")
    print("📍 Application will be available at: http://127.0.0.1:5000")
    print("🔄 Press Ctrl+C to stop the application")
    print("-" * 50)
    app.run(host="0.0.0.0", port=5000, debug=True)
