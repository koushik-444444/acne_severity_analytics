"""
Stage 2: MediaPipe-Based Geometric Region Extractor

Uses MediaPipe FaceLandmarker (Tasks API, 468+ landmarks) to define geometric
boundaries for facial regions that BiSeNet does not segment separately:
- Forehead (above eyebrows to hairline)
- Left cheek (left of nose, between eye and mouth)
- Right cheek (right of nose, between eye and mouth)
- Chin (below mouth to jaw bottom)

These geometric regions are intersected with the BiSeNet skin mask
in the RegionCombiner to produce final pixel-accurate region masks.

Replaces the original dlib-based implementation. MediaPipe advantages:
- No compilation (dlib requires CMake + C++ compiler)
- 468 landmarks vs 68 (7x more precise)
- Already used in the companion acne_grading_project
"""

import os
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import (
        FaceLandmarker,
        FaceLandmarkerOptions,
    )
    _MEDIAPIPE_AVAILABLE = True
except ImportError:
    _MEDIAPIPE_AVAILABLE = False

try:
    from scipy.spatial import ConvexHull
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

# Default path to the FaceLandmarker .task model file
_DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "weights",
    "face_landmarker.task",
)


class LandmarkRegionExtractor:
    """
    Defines facial sub-regions using MediaPipe FaceMesh 468-point landmarks.

    Key MediaPipe landmark indices used (subject's perspective):
        Face oval: 10, 338, 297, 332, ..., 109 (36 points, clockwise)
        Right eyebrow: 70, 63, 105, 66, 107
        Left eyebrow: 300, 293, 334, 296, 336
        Right eye bottom: 145, 153, 154, 155
        Left eye bottom: 374, 380, 381, 382
        Nose bridge: 6, 197, 195, 5, 4
        Nose tip: 1
        Nostrils: 48 (right), 278 (left)
        Mouth: 61 (right corner), 291 (left corner), 0 (top), 17 (bottom)
        Jaw: 234 (right ear), 454 (left ear), 152 (chin bottom)

    Convention: 'right' = subject's right = viewer's left.
    """

    # Eyebrow landmarks (subject's perspective)
    RIGHT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
    LEFT_EYEBROW = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]

    # Eye bottom contours (for computing cheek top boundary)
    RIGHT_EYE_BOTTOM = [145, 153, 154, 155]
    LEFT_EYE_BOTTOM = [374, 380, 381, 382]

    # Nose landmarks (bridge + tip + nostrils)
    NOSE_IDX = [
        6, 197, 195, 5, 4, 1, 2, 98, 327,
        48, 115, 131, 134, 51, 281, 363, 360, 344, 278,
    ]

    # Outer mouth contour
    MOUTH_OUTER = [
        61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
        291, 375, 321, 405, 314, 17, 84, 181, 91, 146,
    ]

    # Face oval / jawline indices (clockwise)
    JAWLINE_IDX = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454]

    def __init__(self, model_path: Optional[str] = None, **kwargs):
        """
        Initialize MediaPipe FaceLandmarker (Tasks API).

        Args:
            model_path: Path to ``face_landmarker.task`` model file.
                Defaults to ``weights/face_landmarker.task`` relative to
                the project root.
            **kwargs: Accepted for API compatibility (e.g. predictor_path
                from the old dlib interface is silently ignored).
        """
        if not _MEDIAPIPE_AVAILABLE:
            raise ImportError(
                "mediapipe is required for landmark extraction. "
                "Install with: pip install mediapipe"
            )

        resolved_path = model_path or _DEFAULT_MODEL_PATH
        if not os.path.isfile(resolved_path):
            raise FileNotFoundError(
                f"FaceLandmarker model not found: {resolved_path}\n"
                "Download from: https://developers.google.com/mediapipe/"
                "solutions/vision/face_landmarker"
            )

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=resolved_path),
            num_faces=1,
        )
        self._landmarker = FaceLandmarker.create_from_options(options)

    def detect_landmarks(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect 468 facial landmarks in a BGR image.

        Args:
            image: BGR image (OpenCV format).

        Returns:
            (468, 2) array of (x, y) pixel coordinates, or None if no face.
        """
        H, W = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb,
        )
        result = self._landmarker.detect(mp_image)

        if not result.face_landmarks or len(result.face_landmarks) == 0:
            return None

        face_lms = result.face_landmarks[0]
        landmarks = np.array(
            [(int(lm.x * W), int(lm.y * H)) for lm in face_lms[:468]],
            dtype=np.int32,
        )
        return landmarks

    def get_region_polygons(
        self, landmarks: np.ndarray, image_shape: Tuple[int, int]
    ) -> Dict[str, np.ndarray]:
        """
        Compute polygon boundaries for each facial region from landmarks.

        Regions (subject's perspective):
        - nose: Convex hull of nose bridge + nostrils
        - right_cheek: Between right jaw, right eye bottom, nose, mouth
        - left_cheek: Between left jaw, left eye bottom, nose, mouth
        - forehead: Above eyebrows to estimated hairline
        - chin: Below mouth to jaw bottom

        Args:
            landmarks: (468, 2) landmark array from detect_landmarks().
            image_shape: (H, W) of the original image.

        Returns:
            Dict mapping region name to (N, 2) polygon array.
        """
        H, W = image_shape

        # --- Key reference points ---
        nose_midx = landmarks[1][0]  # Nose tip X = face midline

        # Eye bottoms
        reye_bottom = max(landmarks[i][1] for i in self.RIGHT_EYE_BOTTOM)
        leye_bottom = max(landmarks[i][1] for i in self.LEFT_EYE_BOTTOM)

        # Mouth top and bottom
        mouth_top = min(landmarks[i][1] for i in self.MOUTH_OUTER)
        mouth_bottom = max(landmarks[i][1] for i in self.MOUTH_OUTER)

        # Eyebrow tops
        rbrow_top = min(landmarks[i][1] for i in self.RIGHT_EYEBROW)
        lbrow_top = min(landmarks[i][1] for i in self.LEFT_EYEBROW)
        brow_top = min(rbrow_top, lbrow_top)

        # Jaw reference points (subject's perspective)
        jaw_right = landmarks[234]   # Subject's right (viewer's left)
        jaw_left = landmarks[454]    # Subject's left (viewer's right)
        jaw_bottom = landmarks[152]  # Chin bottom

        # --- Yaw Awareness (Orientation detection) ---
        # Calculate horizontal distances from nose tip to jaw edges
        dist_right = abs(nose_midx - jaw_right[0])
        dist_left = abs(nose_midx - jaw_left[0])
        total_dist = dist_right + dist_left
        
        # Yaw ratio: < 0.5 means facing left, > 0.5 means facing right
        # We'll use this to adjust cheek boundaries for profile views
        yaw_ratio = dist_right / total_dist if total_dist > 0 else 0.5
        
        # Adjustment factors: reduce cheek width on the "compressed" side
        lcheek_width_mod = 1.0
        rcheek_width_mod = 1.0
        
        if yaw_ratio < 0.4: # Facing significantly left (viewer's)
            rcheek_width_mod = 0.8 # Compress viewer's left cheek
        elif yaw_ratio > 0.6: # Facing significantly right (viewer's)
            lcheek_width_mod = 0.8 # Compress viewer's right cheek

        # Nose nostrils
        nose_right = landmarks[48]   # Right nostril
        nose_left = landmarks[278]   # Left nostril

        # Mouth corners
        mouth_right = landmarks[61]  # Subject's right
        mouth_left = landmarks[291]  # Subject's left

        # --- Nose ---
        nose_pts = landmarks[self.NOSE_IDX]

        # --- Right Cheek (subject's right = viewer's left) ---
        # Ordering points clockwise to form a valid non-self-intersecting polygon
        # Points follow the boundary: Ear -> Jaw -> Chin -> Mouth -> Nose -> Eye -> Ear
        rcheek_pts = np.array([
            landmarks[234],                           # 1. Near ear (upper jaw start)
            landmarks[93],                            # 2. Middle jaw
            landmarks[132],                           # 3. Lower jaw
            landmarks[58],                            # 4. Jaw near chin
            landmarks[172],                           # 5. Jaw near chin
            landmarks[150],                           # 6. Very near chin
            [mouth_right[0], mouth_bottom],           # 7. Bottom mouth corner
            [mouth_right[0], mouth_top],              # 8. Top mouth corner
            landmarks[116],                           # 9. Lower nose side
            nose_right,                               # 10. Nostril
            landmarks[196],                           # 11. Upper nose side
            [landmarks[145][0], reye_bottom],         # 12. Below eye
        ], dtype=np.int32)
        
        # Apply yaw-based width compression if needed (scale X relative to nose_midx)
        if rcheek_width_mod < 1.0:
            rcheek_pts[:, 0] = nose_midx - (nose_midx - rcheek_pts[:, 0]) * rcheek_width_mod

        # --- Left Cheek (subject's left = viewer's right) ---
        # Ordering points counter-clockwise to form a valid non-self-intersecting polygon
        # Points follow the boundary: Ear -> Jaw -> Chin -> Mouth -> Nose -> Eye -> Ear
        lcheek_pts = np.array([
            landmarks[454],                           # 1. Near ear (upper jaw end)
            landmarks[323],                           # 2. Middle jaw
            landmarks[361],                           # 3. Lower jaw
            landmarks[288],                           # 4. Jaw near chin
            landmarks[397],                           # 5. Jaw near chin
            landmarks[379],                           # 6. Very near chin
            [mouth_left[0], mouth_bottom],            # 7. Bottom mouth corner
            [mouth_left[0], mouth_top],               # 8. Top mouth corner
            landmarks[345],                           # 9. Lower nose side
            nose_left,                                # 10. Nostril
            landmarks[419],                           # 11. Upper nose side
            [landmarks[374][0], leye_bottom],         # 12. Below eye
        ], dtype=np.int32)

        # Apply yaw-based width compression if needed
        if lcheek_width_mod < 1.0:
            lcheek_pts[:, 0] = nose_midx + (lcheek_pts[:, 0] - nose_midx) * lcheek_width_mod


        # --- Forehead ---
        # Adaptive height based on eye-to-mouth distance (clinical rule of thirds)
        eye_center_y = (reye_bottom + leye_bottom) / 2
        eye_mouth_dist = mouth_bottom - eye_center_y
        
        # Forehead is usually 0.8 to 1.0 times the eye-mouth distance
        forehead_height = eye_mouth_dist * 0.95
        forehead_top = max(0, int(brow_top - forehead_height))

        forehead_pts = np.array([
            [jaw_right[0] - 10, brow_top],            # Right edge at brow
            [jaw_right[0] - 10, forehead_top],         # Right edge at hairline
            [nose_midx, max(0, forehead_top - 20)],   # Center hairline
            [jaw_left[0] + 10, forehead_top],          # Left edge at hairline
            [jaw_left[0] + 10, brow_top],              # Left edge at brow
            # Left eyebrow (subject's left)
            landmarks[300],
            landmarks[293],
            landmarks[334],
            landmarks[296],
            landmarks[336],
            # Bridge
            landmarks[9],
            # Right eyebrow (subject's right)
            landmarks[107],
            landmarks[66],
            landmarks[105],
            landmarks[63],
            landmarks[70],
        ], dtype=np.int32)

        # --- Chin ---
        chin_pts = np.array([
            landmarks[136],                           # Jaw right side
            landmarks[150],
            landmarks[149],
            landmarks[148],
            landmarks[152],                           # Chin tip
            landmarks[176],
            landmarks[377],                           # Jaw left side
            landmarks[400],
            landmarks[378],
            [mouth_left[0], mouth_bottom],            # Below left mouth corner
            [landmarks[17][0], mouth_bottom],         # Below center mouth
            [mouth_right[0], mouth_bottom],           # Below right mouth corner
        ], dtype=np.int32)

        # --- Jawline/Upper Neck ---
        # The region just below the jawline landmarks
        jaw_pts = landmarks[self.JAWLINE_IDX]
        # Project down slightly to capture the "U-Zone" upper neck
        neck_projected = jaw_pts.copy()
        neck_projected[:, 1] += int((jaw_bottom[1] - mouth_bottom) * 0.8)
        
        # Combine into a strip below the jaw
        jawline_neck_pts = np.vstack([jaw_pts, neck_projected[::-1]])

        # --- Center Line (for strict lateral split) ---
        # Nose bridge + philtrum center
        center_line_pts = landmarks[[6, 197, 195, 5, 4, 1, 2, 0, 17, 152]]

        return {
            "nose": nose_pts,
            "right_cheek": rcheek_pts,
            "left_cheek": lcheek_pts,
            "forehead": forehead_pts,
            "chin": chin_pts,
            "jawline_neck": jawline_neck_pts,
            "center_line": center_line_pts
        }

    def get_region_masks(
        self,
        landmarks: np.ndarray,
        image_shape: Tuple[int, int],
    ) -> Dict[str, np.ndarray]:
        """
        Generate binary masks for each facial region using landmark geometry.

        Args:
            landmarks: (468, 2) array of landmark coordinates.
            image_shape: (H, W) of the target image.

        Returns:
            Dict mapping region name to binary mask (H, W), uint8 (0 or 255).
        """
        H, W = image_shape
        polygons = self.get_region_polygons(landmarks, image_shape)

        masks = {}
        for name, pts in polygons.items():
            mask = np.zeros((H, W), dtype=np.uint8)
            # Use convex hull for robust filling of non-ordered polygons
            try:
                if _SCIPY_AVAILABLE and len(pts) >= 3:
                    hull = ConvexHull(pts)
                    hull_pts = pts[hull.vertices]
                else:
                    hull_pts = pts
            except Exception:
                hull_pts = pts

            cv2.fillPoly(mask, [hull_pts], (255,))
            masks[name] = mask

        return masks
