import os
import urllib.request
from ultralytics import YOLO

def download_models():
    # Use /tmp for everything to avoid permission issues during build
    model_name = "yolov8n-face.pt"
    # Destination in the app folder
    app_path = os.path.join(os.getcwd(), model_name)
    
    # This is a common community source for the YOLOv8-face model
    url = "https://github.com/derronqi/yolov8-face/releases/download/v1.0/yolov8n-face.pt"
    
    print(f"Target path: {app_path}")
    
    if not os.path.exists(app_path):
        print(f"Downloading {model_name} from {url}...")
        try:
            # Set a user-agent to avoid potential blocks
            opener = urllib.request.build_opener()
            opener.addheaders = [('User-agent', 'Mozilla/5.0')]
            urllib.request.install_opener(opener)
            
            urllib.request.urlretrieve(url, app_path)
            print("Download complete.")
        except Exception as e:
            print(f"Failed to download face model: {e}")
            print("Attempting to load standard yolov8n.pt as fallback...")
            # Ultralytics will auto-download this if it succeeds in reaching their server
            try:
                model = YOLO("yolov8n.pt")
                print("Standard YOLOv8n cached.")
                return
            except Exception as e2:
                print(f"Fallback also failed: {e2}")
                raise e

    print(f"Verifying {app_path}...")
    try:
        # Load once to ensure it's valid and cached
        model = YOLO(app_path)
        print("Model verified successfully.")
    except Exception as e:
        print(f"Error loading model from {app_path}: {e}")
        raise e

if __name__ == "__main__":
    download_models()
