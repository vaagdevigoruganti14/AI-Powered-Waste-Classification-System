#!/usr/bin/env python3
"""
Smart Waste Classifier - Startup Script
This script provides an easy way to start the waste classification application.
"""

import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path

def check_requirements():
    """Check if all required files exist"""
    required_files = [
        "app.py",
        "waste_classifier_cnn.h5", 
        "class_indices.json",
        "index.html"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    print("✅ All required files found")
    return True

def check_dependencies():
    """Check if dependencies are installed"""
    try:
        import flask
        import tensorflow as tf
        import numpy as np
        from PIL import Image
        print("✅ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def start_application():
    """Start the Flask application"""
    print("\n🚀 Starting Smart Waste Classifier...")
    print("📍 Application will be available at: http://127.0.0.1:5000")
    print("🔄 Press Ctrl+C to stop the application")
    print("-" * 50)
    
    try:
        # Start the Flask app
        from app import app
        app.run(host="0.0.0.0", port=5000, debug=True)
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting application: {e}")
        return False
    
    return True

def main():
    """Main startup function"""
    print("♻️  Smart Waste Classifier")
    print("=" * 40)
    
    # Check if we're in the right directory
    if not os.path.exists("app.py"):
        print("❌ Please run this script from the project root directory")
        sys.exit(1)
    
    # Check requirements
    if not check_requirements():
        print("\n💡 Make sure you have trained the model first:")
        print("   python train.py")
        sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Start the application
    if start_application():
        print("\n✅ Application started successfully!")
    else:
        print("\n❌ Failed to start application")
        sys.exit(1)

if __name__ == "__main__":
    main()

