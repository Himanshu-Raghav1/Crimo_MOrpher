"""
Flask Web Application for Face Morphing.
Captures/uploads photos, detects faces with YOLO, and applies morph effects.
"""

import os
import base64
import time

import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify

from face_detector import FaceDetector
from landmark_detector import LandmarkDetector
from morph_engine import MorphEngine
from bg_replacer import BackgroundReplacer

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max upload

# Initialize modules
print("Loading face detector...")
face_detector = FaceDetector()
print("Loading landmark detector...")
landmark_detector = LandmarkDetector()
print("Loading morph engine...")
morph_engine = MorphEngine()
print("Loading background replacer...")
bg_replacer = BackgroundReplacer()
print("All models loaded!")

# Output directory
os.makedirs("output", exist_ok=True)


def decode_image(data_url):
    """Decode base64 data URL to OpenCV image."""
    if "," in data_url:
        data_url = data_url.split(",")[1]
    img_bytes = base64.b64decode(data_url)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return image


def encode_image(image, fmt=".jpg"):
    """Encode OpenCV image to base64 data URL."""
    _, buffer = cv2.imencode(fmt, image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    b64 = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


@app.route("/")
def index():
    """Serve the main web UI."""
    effects = morph_engine.get_effect_names()
    return render_template("index.html", effects=effects)


@app.route("/detect", methods=["POST"])
def detect():
    """Detect faces in the uploaded image."""
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "No image provided"}), 400

    image = decode_image(data["image"])
    if image is None:
        return jsonify({"error": "Could not decode image"}), 400

    # Detect faces
    faces = face_detector.detect(image)
    if not faces:
        return jsonify({"error": "No face detected. Try better lighting or angle."}), 200

    # Draw detections
    annotated = face_detector.draw_detections(image, faces)

    # Get landmarks for first face
    bbox = faces[0]["bbox"]
    landmarks = landmark_detector.get_landmarks(image, bbox)

    result = {
        "annotated_image": encode_image(annotated),
        "faces_count": len(faces),
        "face_bbox": list(bbox),
        "has_landmarks": landmarks is not None,
    }

    return jsonify(result)


@app.route("/morph", methods=["POST"])
def morph():
    """Apply morphing effect to the image."""
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "No image provided"}), 400

    try:
        effect = data.get("effect", "bulge")
        strength = float(data.get("strength", 1.0))
        criminal_bg = bool(data.get("criminal_bg", False))

        image = decode_image(data["image"])
        if image is None:
            return jsonify({"error": "Could not decode image"}), 400

        # Detect face
        faces = face_detector.detect(image)
        if not faces:
            return jsonify({"error": "No face detected. Make sure your face is clearly visible."})

        bbox = faces[0]["bbox"]

        # Get landmarks — always returns valid array (never None)
        landmarks = landmark_detector.get_landmarks(image, bbox)

        # Apply morph effect
        morphed = morph_engine.apply(image, landmarks, effect, strength)

        # Optionally replace background with criminal mugshot scene
        if criminal_bg:
            morphed = bg_replacer.apply(morphed, bbox)

        result = {
            "morphed_image": encode_image(morphed),
            "effect": effect,
            "strength": strength,
            "criminal_bg": criminal_bg,
        }
        return jsonify(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Morph failed: {str(e)}"}), 500


@app.route("/save", methods=["POST"])
def save():
    """Save the morphed image to disk."""
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "No image provided"}), 400

    image = decode_image(data["image"])
    if image is None:
        return jsonify({"error": "Could not decode image"}), 400

    timestamp = int(time.time())
    effect = data.get("effect", "unknown")
    filename = f"morph_{effect}_{timestamp}.jpg"
    filepath = os.path.join("output", filename)

    cv2.imwrite(filepath, image)

    return jsonify({
        "success": True,
        "filename": filename,
        "path": os.path.abspath(filepath),
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(debug=False, host="0.0.0.0", port=port, use_reloader=False)
