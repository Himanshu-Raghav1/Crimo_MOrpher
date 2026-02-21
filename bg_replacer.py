"""
Background Replacer Module — Improved Edge Detection.
Uses multi-pass GrabCut + Canny edge refinement for clean person cutouts.
"""

import cv2
import numpy as np
import datetime


class BackgroundReplacer:

    def __init__(self):
        self._cached_bg = None
        self._cached_size = None

    def apply(self, image, face_bbox):
        """Replace background with a police mugshot/lineup background."""
        h, w = image.shape[:2]
        bg = self._make_mugshot_bg(w, h)
        mask_soft = self._segment_person(image, face_bbox)  # float 0.0–1.0
        # Composite: blend person over bg using soft alpha mask
        mask3 = mask_soft[:, :, np.newaxis]
        result = (image.astype(np.float32) * mask3 +
                  bg.astype(np.float32) * (1.0 - mask3))
        result = np.clip(result, 0, 255).astype(np.uint8)
        return self._add_vignette(result)

    # ------------------------------------------------------------------
    # Police Lineup Background Generator
    # ------------------------------------------------------------------

    def _make_mugshot_bg(self, w, h):
        if self._cached_bg is not None and self._cached_size == (w, h):
            return self._cached_bg.copy()

        bg = np.zeros((h, w, 3), dtype=np.uint8)

        # Dark blue-gray gradient wall
        for y in range(h):
            t = y / h
            bg[y, :] = (int(30 + t * 20), int(40 + t * 20), int(70 + t * 30))

        # Height ruler marks: 4'0" to 7'0"
        ruler_top = int(h * 0.05)
        ruler_bot = int(h * 0.90)
        ruler_range = ruler_bot - ruler_top
        total_inches = 36  # 7'0" - 4'0"

        for ft in range(4, 8):
            for half in [0, 6]:
                total_in = ft * 12 + half - 48
                y_pos = ruler_bot - int((total_in / total_inches) * ruler_range)
                if y_pos < 0 or y_pos >= h:
                    continue
                is_major = (half == 0)
                color = (200, 210, 230) if is_major else (130, 145, 165)
                thick = 2 if is_major else 1
                cv2.line(bg, (0, y_pos), (w, y_pos), color, thick)
                label = f"{ft}'" + (f'{half}"' if half else '0"')
                fs = 0.55 if is_major else 0.38
                cv2.putText(bg, label, (8, y_pos - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, fs, color, 1, cv2.LINE_AA)
                (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
                cv2.putText(bg, label, (w - tw - 8, y_pos - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, fs, color, 1, cv2.LINE_AA)

        # Header bar
        hdr_h = max(28, int(h * 0.065))
        cv2.rectangle(bg, (0, 0), (w, hdr_h), (12, 18, 45), -1)
        cv2.putText(bg, "METROPOLITAN POLICE DEPARTMENT",
                    (int(w * 0.08), hdr_h // 2 + 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (210, 220, 255), 2, cv2.LINE_AA)

        # Footer bar
        ftr_y = int(h * 0.92)
        cv2.rectangle(bg, (0, ftr_y), (w, h), (12, 18, 45), -1)
        today = datetime.date.today().strftime("%Y-%m-%d")
        case_no = str(np.random.randint(1000, 9999))
        footer = f"CASE#: {case_no}    DATE: {today}    BOOKING"
        cv2.putText(bg, footer, (10, ftr_y + int((h - ftr_y) * 0.65)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (170, 185, 210), 1, cv2.LINE_AA)

        self._cached_bg = bg.copy()
        self._cached_size = (w, h)
        return bg

    # ------------------------------------------------------------------
    # Improved Person Segmentation
    # ------------------------------------------------------------------

    def _segment_person(self, image, face_bbox):
        """
        Returns a soft float mask (H x W), values 0.0–1.0.
        1.0 = definitely person, 0.0 = definitely background.
        Uses multi-pass GrabCut + Canny edge refinement + feathering.
        """
        h, w = image.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in face_bbox]
        fw = x2 - x1
        fh = y2 - y1

        # ── Step 1: Estimate full-body bounding box ─────────────────
        # Head is roughly 1/7 of body height; expand generously
        body_top    = max(0,    y1 - int(fh * 0.5))
        body_bot    = min(h,    y2 + int(fh * 6.5))
        body_left   = max(0,    x1 - int(fw * 1.8))
        body_right  = min(w,    x2 + int(fw * 1.8))

        rect = (body_left, body_top,
                body_right - body_left, body_bot - body_top)

        if rect[2] < 2 or rect[3] < 2:
            return np.ones((h, w), dtype=np.float32)

        # ── Step 2: Pass 1 GrabCut (5 iterations, rect mode) ────────
        mask_gc = np.zeros((h, w), np.uint8)
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)

        try:
            cv2.grabCut(image, mask_gc, rect, bgd, fgd, 5,
                        cv2.GC_INIT_WITH_RECT)
        except Exception:
            # Fallback: rectangular region as foreground
            mask_gc[body_top:body_bot, body_left:body_right] = cv2.GC_PR_FGD

        person_bin = np.isin(mask_gc, [cv2.GC_FGD, cv2.GC_PR_FGD]).astype(np.uint8)

        # ── Step 3: Seed face area as definite foreground, then re-run
        face_pad = int(min(fw, fh) * 0.15)
        fy1 = max(0, y1 - face_pad)
        fy2 = min(h, y2 + face_pad)
        fx1 = max(0, x1 - face_pad)
        fx2 = min(w, x2 + face_pad)
        mask_gc[fy1:fy2, fx1:fx2] = cv2.GC_FGD  # force face as FG

        # 2-pixel border of the bbox as definite background
        border = 4
        mask_gc[:border, :] = cv2.GC_BGD
        mask_gc[-border:, :] = cv2.GC_BGD
        mask_gc[:, :border] = cv2.GC_BGD
        mask_gc[:, -border:] = cv2.GC_BGD

        try:
            cv2.grabCut(image, mask_gc, None, bgd, fgd, 3,
                        cv2.GC_INIT_WITH_MASK)
        except Exception:
            pass

        person_bin = np.isin(mask_gc, [cv2.GC_FGD, cv2.GC_PR_FGD]).astype(np.uint8) * 255

        # ── Step 4: Canny edge refinement ───────────────────────────
        # Use edges to sharpen the mask boundary
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 100)

        # Dilate edges slightly, then use as a boundary hint
        edge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges_dilated = cv2.dilate(edges, edge_kernel, iterations=1)

        # Where there's an edge near the mask boundary, trust the edge
        # (zero out uncertain pixels near edges that GrabCut left as BG)
        boundary_zone = cv2.dilate(person_bin, edge_kernel, iterations=6) - \
                        cv2.erode(person_bin, edge_kernel, iterations=6)
        # In the boundary zone, if there's a strong edge, include that pixel
        person_bin[np.logical_and(boundary_zone > 0, edges_dilated > 0)] = 255

        # ── Step 5: Morphological cleanup ───────────────────────────
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        person_bin = cv2.morphologyEx(person_bin, cv2.MORPH_CLOSE, k_close, iterations=2)
        person_bin = cv2.morphologyEx(person_bin, cv2.MORPH_OPEN,  k_open,  iterations=1)

        # Keep only the largest connected component (the person)
        person_bin = self._keep_largest_blob(person_bin)

        # ── Step 6: Soft feathering at edges ────────────────────────
        # Gaussian blur the binary mask → smooth alpha
        soft_mask = cv2.GaussianBlur(person_bin.astype(np.float32) / 255.0,
                                     (21, 21), 0)
        # Re-sharpen the centre to avoid see-through body
        soft_mask = np.clip(soft_mask * 1.4, 0.0, 1.0)

        return soft_mask

    # ------------------------------------------------------------------

    def _keep_largest_blob(self, binary_mask):
        """Keep only the largest connected white region (the person)."""
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary_mask, connectivity=8)
        if num_labels <= 1:
            return binary_mask
        # Ignore label 0 (background); find label with largest area
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        result = np.zeros_like(binary_mask)
        result[labels == largest] = 255
        return result

    # ------------------------------------------------------------------
    # Vignette
    # ------------------------------------------------------------------

    def _add_vignette(self, image):
        h, w = image.shape[:2]
        gy = np.linspace(0, 1, h).reshape(-1, 1)
        gx = np.linspace(0, 1, w).reshape(1, -1)
        vig = np.clip((4 * gy * (1 - gy)) * (4 * gx * (1 - gx)) * 1.15,
                      0, 1).astype(np.float32)
        out = image.astype(np.float32) * vig[:, :, np.newaxis]
        return np.clip(out, 0, 255).astype(np.uint8)
