"""
Facial Landmark Detection — MediaPipe primary, geometric fallback.
Always returns valid 68 points, never None.
"""

import cv2
import numpy as np

_MP_OK = False
try:
    import mediapipe as mp
    _MP_OK = True
except ImportError:
    pass


class LandmarkDetector:

    def __init__(self):
        self.mp_face_mesh = None

        if _MP_OK:
            try:
                self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                )
                print("LandmarkDetector: using MediaPipe FaceMesh")
            except Exception as e:
                print(f"LandmarkDetector: MediaPipe init failed ({e}), using geometric fallback")
        else:
            print("LandmarkDetector: using geometric fallback")

    def get_landmarks(self, image, bbox=None):
        """Always returns (68, 2) numpy array. Never None."""
        if self.mp_face_mesh is not None:
            result = self._mediapipe(image, bbox)
            if result is not None:
                return result
        return self._geometric(image, bbox)

    # ── MediaPipe ─────────────────────────────────────────────────────

    def _mediapipe(self, image, bbox=None):
        try:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.mp_face_mesh.process(rgb)
            if not results.multi_face_landmarks:
                return None

            face = results.multi_face_landmarks[0]
            h, w = image.shape[:2]

            mp_to_68 = [
                10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,
                70,63,105,66,107,
                336,296,334,293,300,
                168,6,197,195,5,4,1,19,94,
                33,160,158,133,153,144,
                362,385,387,263,373,380,
                61,39,37,0,267,269,291,405,314,17,84,181,
                78,82,13,312,308,317,14,87,
            ]

            pts = []
            for idx in mp_to_68:
                if idx < len(face.landmark):
                    lm = face.landmark[idx]
                    pts.append((int(lm.x * w), int(lm.y * h)))
                else:
                    pts.append((w // 2, h // 2))

            return np.array(pts, dtype=np.int32)
        except Exception:
            return None

    # ── Geometric fallback ────────────────────────────────────────────

    def _geometric(self, image, bbox=None):
        h_img, w_img = image.shape[:2]

        if bbox is None:
            bbox = (int(w_img * 0.15), int(h_img * 0.15),
                    int(w_img * 0.85), int(h_img * 0.85))

        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = max(x1+1, min(x2, w_img)), max(y1+1, min(y2, h_img))

        fw, fh = x2 - x1, y2 - y1
        cx, cy = x1 + fw // 2, y1 + fh // 2
        pts = []

        # Jaw (17)
        for i in range(17):
            t = i / 16.0
            a = np.pi * 0.1 + t * np.pi * 0.8
            pts.append((int(cx + fw*0.48*np.cos(a)), int(cy + fh*0.50*np.sin(a))))

        # Right eyebrow (5)
        for i in range(5):
            pts.append((int(x1 + fw*(0.18 + i/4.0*0.22)), int(y1 + fh*0.28)))

        # Left eyebrow (5)
        for i in range(5):
            pts.append((int(x1 + fw*(0.60 + i/4.0*0.22)), int(y1 + fh*0.28)))

        # Nose (9)
        for i in range(9):
            t = i / 8.0
            pts.append((int(cx + fw*0.04*np.sin(t*np.pi)), int(y1 + fh*(0.38 + t*0.22))))

        # Right eye (6)
        ecx_r, ecy = int(x1 + fw*0.30), int(y1 + fh*0.38)
        for i in range(6):
            a = (i/6.0)*2*np.pi
            pts.append((int(ecx_r + fw*0.07*np.cos(a)), int(ecy + fh*0.035*np.sin(a))))

        # Left eye (6)
        ecx_l = int(x1 + fw*0.70)
        for i in range(6):
            a = (i/6.0)*2*np.pi
            pts.append((int(ecx_l + fw*0.07*np.cos(a)), int(ecy + fh*0.035*np.sin(a))))

        # Mouth outer (12)
        mcy = int(y1 + fh * 0.72)
        for i in range(12):
            a = (i/12.0)*2*np.pi
            pts.append((int(cx + fw*0.16*np.cos(a)), int(mcy + fh*0.05*np.sin(a))))

        # Mouth inner (8)
        for i in range(8):
            a = (i/8.0)*2*np.pi
            pts.append((int(cx + fw*0.10*np.cos(a)), int(mcy + fh*0.03*np.sin(a))))

        arr = np.array(pts[:68], dtype=np.int32)
        arr[:, 0] = np.clip(arr[:, 0], 0, w_img - 1)
        arr[:, 1] = np.clip(arr[:, 1], 0, h_img - 1)
        return arr
