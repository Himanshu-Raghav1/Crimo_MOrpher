import os
import urllib.request
from ultralytics import YOLO

def download_models():
    model_name = "yolov8n-face.pt"
    # This is a common community source for the YOLOv8-face model
    url = "https://github.com/derronqi/yolov8-face/releases/download/v1.0/yolov8n-face.pt"
    
    print(f"Checking for {model_name}...")
    if not os.path.exists(model_name):
        print(f"Downloading {model_name} from {url}...")
        try:
            urllib.request.urlretrieve(url, model_name)
            print("Download complete.")
        except Exception as e:
            print(f"Failed to download model: {e}")
            # Fallback to standard yolov8n.pt if face-specific model fails
            print("Trying fallback to standard yolov8n.pt...")
            model_name = "yolov8n.pt"

    print(f"Caching {model_name}...")
    try:
        model = YOLO(model_name)
        print("Model cached successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")

if __name__ == "__main__":
    download_models()
