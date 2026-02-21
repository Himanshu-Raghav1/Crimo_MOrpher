from ultralytics import YOLO
import os

def download_models():
    print("Pre-downloading YOLOv8 face model...")
    os.makedirs("models", exist_ok=True)
    # This will download the model to the current directory if not present
    # We force it to load it once to ensure it's cached
    model = YOLO("yolov8n-face.pt")
    print("YOLOv8 face model downloaded successfully.")

if __name__ == "__main__":
    download_models()
