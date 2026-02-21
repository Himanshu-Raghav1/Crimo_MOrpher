"""
Face Detection module — YOLOv8 primary, Haar cascade fallback.
"""

import cv2
import numpy as np

try:
    from ultralytics import YOLO
    _YOLO_OK = True
except ImportError:
    _YOLO_OK = False


class FaceDetector:
    def __init__(self):
        self.model = None
        self.cascade = None

        if _YOLO_OK:
            try:
                self.model = YOLO("yolov8n-face.pt")
                print("FaceDetector: using YOLOv8")
            except Exception:
                pass

        if self.model is None:
            self.cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            print("FaceDetector: using Haar cascade fallback")

    def detect(self, image):
        faces = []

        if self.model is not None:
            results = self.model(image, verbose=False)
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    confidence = float(box.conf[0])
                    faces.append({
                        "bbox": (int(x1), int(y1), int(x2), int(y2)),
                        "confidence": confidence,
                    })
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            detections = self.cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
            )
            for (x, y, w, h) in detections:
                faces.append({
                    "bbox": (int(x), int(y), int(x + w), int(y + h)),
                    "confidence": 1.0,
                })

        return faces

    def draw_detections(self, image, faces):
        annotated = image.copy()
        for face in faces:
            x1, y1, x2, y2 = face["bbox"]
            conf = face["confidence"]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 128), 2)
            label = f"Face {conf:.2f}"
            cv2.putText(annotated, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 128), 2)
        return annotated
