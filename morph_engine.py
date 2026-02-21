"""
Morphing Engine - applies various morphing effects to face images.
Uses facial landmarks and OpenCV for transformations.
"""

import cv2
import numpy as np


class MorphEngine:
    """Applies various morphing effects to detected faces."""

    EFFECTS = ["bulge", "cartoon", "squeeze", "big_eyes", "wide_smile"]

    def apply(self, image, landmarks, effect_name, strength=1.0):
        """
        Apply a morphing effect.

        Args:
            image: numpy array (BGR)
            landmarks: numpy array of shape (68, 2)
            effect_name: one of EFFECTS
            strength: effect intensity (0.0 - 2.0)

        Returns:
            morphed image (numpy array)
        """
        if landmarks is None:
            return image.copy()

        effect_map = {
            "bulge": self._bulge,
            "cartoon": self._cartoon,
            "squeeze": self._squeeze,
            "big_eyes": self._big_eyes,
            "wide_smile": self._wide_smile,
        }

        func = effect_map.get(effect_name, self._bulge)
        return func(image, landmarks, strength)

    def _bulge(self, image, landmarks, strength=1.0):
        """
        Bulge effect - expands face outward from center.
        Creates a fish-eye / inflated look.
        """
        result = image.copy()
        h, w = image.shape[:2]

        # Face center and radius from landmarks
        face_center = landmarks.mean(axis=0).astype(int)
        cx, cy = face_center

        # Calculate face radius from jaw landmarks
        jaw = landmarks[:17]
        face_radius = int(np.max(np.linalg.norm(jaw - face_center, axis=1)) * 1.1)

        # Create coordinate maps
        map_x = np.zeros((h, w), dtype=np.float32)
        map_y = np.zeros((h, w), dtype=np.float32)

        for y in range(h):
            for x in range(w):
                map_x[y, x] = x
                map_y[y, x] = y

        # Vectorized bulge computation
        yy, xx = np.mgrid[0:h, 0:w]
        dx = xx - cx
        dy = yy - cy
        dist = np.sqrt(dx * dx + dy * dy)

        mask = dist < face_radius
        norm_dist = dist[mask] / face_radius

        # Bulge function: push pixels outward
        bulge_amount = strength * 0.5
        new_dist = norm_dist ** (1.0 + bulge_amount)
        scale = np.where(norm_dist > 0, new_dist / norm_dist, 1.0)

        map_x[mask] = cx + dx[mask] * scale
        map_y[mask] = cy + dy[mask] * scale

        result = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT)
        return result

    def _cartoon(self, image, landmarks, strength=1.0):
        """
        Cartoon / Comic effect — applies bilateral smoothing for flat color
        regions and adaptive edge lines, creating a hand-drawn cartoon look.
        Strength controls the edge boldness and smoothing intensity.
        """
        # Number of bilateral filter passes scales with strength
        passes = max(1, int(strength * 2))
        d = 9
        sigma_color = int(75 * strength)
        sigma_space = int(75 * strength)

        # Step 1: Smooth colors while preserving edges (bilateral filter)
        smooth = image.copy()
        for _ in range(passes):
            smooth = cv2.bilateralFilter(smooth, d, sigma_color, sigma_space)

        # Step 2: Detect edges on grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.medianBlur(gray, 7)

        # Adaptive threshold gives bold cartoon outlines
        block_size = 9
        edges = cv2.adaptiveThreshold(
            gray_blur, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            block_size,
            int(2 + strength * 3)
        )

        # Step 3: Combine smooth colors with edge mask
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        cartoon = cv2.bitwise_and(smooth, edges_bgr)

        return cartoon

    def _squeeze(self, image, landmarks, strength=1.0):
        """
        Squeeze effect - compresses face horizontally, stretches vertically.
        Creates a tall, thin face look.
        """
        result = image.copy()
        h, w = image.shape[:2]

        face_center = landmarks.mean(axis=0).astype(int)
        cx, cy = face_center

        jaw = landmarks[:17]
        face_radius = int(np.max(np.linalg.norm(jaw - face_center, axis=1)) * 1.1)

        yy, xx = np.mgrid[0:h, 0:w]
        dx = (xx - cx).astype(np.float64)
        dy = (yy - cy).astype(np.float64)
        dist = np.sqrt(dx * dx + dy * dy)

        map_x = xx.astype(np.float32)
        map_y = yy.astype(np.float32)

        mask = dist < face_radius
        norm_dist = dist[mask] / face_radius

        # Squeeze horizontally, stretch vertically
        squeeze = strength * 0.3
        falloff = 1.0 - norm_dist  # stronger at center

        sx = 1.0 + squeeze * falloff   # expand x (pull source from wider)
        sy = 1.0 - squeeze * 0.5 * falloff  # compress y

        map_x[mask] = (cx + dx[mask] * sx).astype(np.float32)
        map_y[mask] = (cy + dy[mask] * sy).astype(np.float32)

        result = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT)
        return result

    def _big_eyes(self, image, landmarks, strength=1.0):
        """
        Big Eyes effect - enlarges the eye regions.
        """
        result = image.copy()
        h, w = image.shape[:2]

        # Get eye centers and radii
        right_eye = landmarks[36:42]
        left_eye = landmarks[42:48]

        for eye_pts in [right_eye, left_eye]:
            eye_center = eye_pts.mean(axis=0).astype(int)
            ex, ey = eye_center

            # Eye radius
            eye_w = np.max(eye_pts[:, 0]) - np.min(eye_pts[:, 0])
            eye_h = np.max(eye_pts[:, 1]) - np.min(eye_pts[:, 1])
            eye_radius = int(max(eye_w, eye_h) * 1.5)

            if eye_radius < 5:
                continue

            # Apply local bulge on each eye
            # Extract a region around the eye
            r = eye_radius + 10
            y1 = max(0, ey - r)
            y2 = min(h, ey + r)
            x1 = max(0, ex - r)
            x2 = min(w, ex + r)

            region = result[y1:y2, x1:x2].copy()
            rh, rw = region.shape[:2]
            rcx = ex - x1
            rcy = ey - y1

            map_x_r = np.zeros((rh, rw), dtype=np.float32)
            map_y_r = np.zeros((rh, rw), dtype=np.float32)

            yy, xx = np.mgrid[0:rh, 0:rw]
            dx = (xx - rcx).astype(np.float64)
            dy = (yy - rcy).astype(np.float64)
            dist = np.sqrt(dx * dx + dy * dy)

            map_x_r = xx.astype(np.float32)
            map_y_r = yy.astype(np.float32)

            mask = dist < eye_radius
            norm_dist = dist[mask] / eye_radius

            # Magnification: map outward pixels to inner region
            magnify = strength * 0.6
            new_dist = norm_dist ** (1.0 + magnify)
            scale = np.where(norm_dist > 0, new_dist / norm_dist, 1.0)

            map_x_r[mask] = (rcx + dx[mask] * scale).astype(np.float32)
            map_y_r[mask] = (rcy + dy[mask] * scale).astype(np.float32)

            warped = cv2.remap(region, map_x_r, map_y_r, cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT)
            result[y1:y2, x1:x2] = warped

        return result

    def _wide_smile(self, image, landmarks, strength=1.0):
        """
        Wide Smile effect - stretches the mouth area horizontally
        and curves it upward.
        """
        result = image.copy()
        h, w = image.shape[:2]

        # Mouth landmarks
        mouth = landmarks[48:68]
        mouth_center = mouth.mean(axis=0).astype(int)
        mx, my = mouth_center

        # Mouth dimensions
        mouth_w = np.max(mouth[:, 0]) - np.min(mouth[:, 0])
        mouth_h = np.max(mouth[:, 1]) - np.min(mouth[:, 1])
        mouth_radius = int(max(mouth_w, mouth_h) * 1.8)

        if mouth_radius < 10:
            return result

        # Area around mouth
        r = mouth_radius + 15
        y1 = max(0, my - r)
        y2 = min(h, my + r)
        x1 = max(0, mx - r)
        x2 = min(w, mx + r)

        region = result[y1:y2, x1:x2].copy()
        rh, rw = region.shape[:2]
        rcx = mx - x1
        rcy = my - y1

        yy, xx = np.mgrid[0:rh, 0:rw]
        dx = (xx - rcx).astype(np.float64)
        dy = (yy - rcy).astype(np.float64)
        dist = np.sqrt(dx * dx + dy * dy)

        map_x_r = xx.astype(np.float32)
        map_y_r = yy.astype(np.float32)

        mask = dist < mouth_radius
        norm_dist = dist[mask] / mouth_radius
        falloff = 1.0 - norm_dist

        # Stretch horizontally
        stretch = strength * 0.3
        sx = 1.0 + stretch * falloff

        # Curve upward at the edges
        curve = strength * 5.0
        curve_offset = -curve * falloff * (dx[mask] / mouth_radius) ** 2

        map_x_r[mask] = (rcx + dx[mask] * sx).astype(np.float32)
        map_y_r[mask] = (rcy + dy[mask] + curve_offset).astype(np.float32)

        warped = cv2.remap(region, map_x_r, map_y_r, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT)
        result[y1:y2, x1:x2] = warped

        return result

    def get_effect_names(self):
        """Return list of available effect names."""
        return [
            {"id": "bulge", "name": "👁 Bulge", "desc": "Fish-eye inflation"},
            {"id": "cartoon", "name": "🎨 Cartoon", "desc": "Comic book style"},
            {"id": "squeeze", "name": "🤏 Squeeze", "desc": "Tall & thin"},
            {"id": "big_eyes", "name": "👀 Big Eyes", "desc": "Enlarged eyes"},
            {"id": "wide_smile", "name": "😁 Wide Smile", "desc": "Exaggerated grin"},
        ]
