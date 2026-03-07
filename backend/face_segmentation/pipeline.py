"""
FaceSegmentationPipeline - Main orchestrator for the two-stage face segmentation.

Stage 1: BiSeNet face parsing (19-class semantic segmentation)
Stage 2: MediaPipe landmark-based geometric partitioning (468-point FaceMesh)

Output: pixel-accurate binary masks for nose, left_cheek, right_cheek,
        forehead, chin.
"""

import os
import time
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from face_segmentation.face_parser import FaceParser
from face_segmentation.landmark_extractor import LandmarkRegionExtractor
from face_segmentation.region_combiner import RegionCombiner


class FaceSegmentationPipeline:
    """
    End-to-end face region segmentation pipeline.

    Usage:
        pipeline = FaceSegmentationPipeline(
            bisenet_weights="weights/79999_iter.pth",
        )
        results = pipeline.segment("path/to/face.jpg")
        # results["masks"] = {"nose": ..., "left_cheek": ..., ...}
    """

    def __init__(
        self,
        bisenet_weights: str = "weights/79999_iter.pth",
        device: Optional[str] = None,
        input_size: Tuple[int, int] = (512, 512),
        morph_iterations: int = 1,
        smooth_edges: bool = False,
        **kwargs,
    ):
        """
        Args:
            bisenet_weights: Path to BiSeNet pretrained weights.
            device: 'cuda', 'cpu', or None for auto-detect.
            input_size: (H, W) input resolution for BiSeNet.
            morph_iterations: Morphological cleanup iterations (0 to disable).
            smooth_edges: If True, apply slight Gaussian blur to mask edges.
            **kwargs: Ignored (for backward compatibility with old dlib args).
        """
        print("[Pipeline] Initializing face segmentation pipeline...")

        # Stage 1: BiSeNet face parser
        print(f"[Stage 1] Loading BiSeNet from: {bisenet_weights}")
        self.parser = FaceParser(
            weight_path=bisenet_weights,
            device=device,
            input_size=input_size,
        )

        # Stage 2: MediaPipe landmark extractor (468-point FaceMesh)
        print("[Stage 2] Initializing MediaPipe FaceMesh...")
        self.landmark_extractor = LandmarkRegionExtractor()

        # Combiner
        self.combiner = RegionCombiner(
            morph_iterations=morph_iterations,
            smooth_edges=smooth_edges
        )

        print("[Pipeline] Ready.")

    def segment(
        self,
        image: np.ndarray,
        return_intermediates: bool = False,
    ) -> Dict:
        """
        Run full segmentation pipeline on a BGR image.

        Args:
            image: BGR image as numpy array, shape (H, W, 3).
            return_intermediates: If True, also return parsing map,
                landmarks, and landmark-only masks for debugging.

        Returns:
            Dict with keys:
                - "masks": Dict[str, np.ndarray] - final region masks
                - "coverage": Dict[str, int] - pixel counts per region
                - "timing": Dict[str, float] - timing info (seconds)
                - "landmarks": np.ndarray (if return_intermediates)
                - "parsing": np.ndarray (if return_intermediates)
                - "landmark_masks": Dict (if return_intermediates)
        """
        result = {}
        timing = {}
        H, W = image.shape[:2]

        # --- Stage 1: BiSeNet Parsing ---
        t0 = time.perf_counter()
        parsing = self.parser.parse(image)
        timing["bisenet"] = time.perf_counter() - t0

        # --- Stage 2: Landmark Detection ---
        t0 = time.perf_counter()
        landmarks = self.landmark_extractor.detect_landmarks(image)
        timing["landmarks"] = time.perf_counter() - t0

        if landmarks is None:
            print("[Warning] No face detected by MediaPipe. Falling back to parsing-only.")
            result["masks"] = self._fallback_parsing_only(parsing, H, W)
            result["coverage"] = RegionCombiner.compute_coverage(result["masks"])
            result["timing"] = timing
            if return_intermediates:
                result["parsing"] = parsing
                result["landmarks"] = None
                result["landmark_masks"] = {}
            return result

        # --- Stage 2: Geometric Region Masks ---
        t0 = time.perf_counter()
        landmark_masks = self.landmark_extractor.get_region_masks(
            landmarks, (H, W)
        )
        timing["geometry"] = time.perf_counter() - t0

        # --- Combination ---
        t0 = time.perf_counter()
        masks = self.combiner.combine(
            parsing=parsing,
            landmark_masks=landmark_masks,
            nose_from_bisenet=True,
        )
        timing["combine"] = time.perf_counter() - t0
        timing["total"] = sum(timing.values())

        result["masks"] = masks
        result["coverage"] = RegionCombiner.compute_coverage(masks)
        
        # Calculate Metadata (JSON serializable)
        from face_segmentation.utils.visualization import calculate_erythema_index
        
        total_pixels = sum(result["coverage"].values())
        result["metadata"] = {
            "total_face_pixels": total_pixels,
            "regions": {
                name: {
                    "pixel_count": count,
                    "area_ratio": count / total_pixels if total_pixels > 0 else 0,
                    "erythema_index": calculate_erythema_index(image, masks[name])
                }
                for name, count in result["coverage"].items()
            },
            "timing_ms": {k: v * 1000 for k, v in timing.items()}
        }
        
        result["timing"] = timing

        if return_intermediates:
            result["parsing"] = parsing
            result["landmarks"] = landmarks
            result["landmark_masks"] = landmark_masks

        return result

    def segment_file(
        self,
        image_path: str,
        return_intermediates: bool = False,
    ) -> Dict:
        """
        Convenience method to segment from a file path.

        Args:
            image_path: Path to image file.
            return_intermediates: Whether to return debug info.

        Returns:
            Same as segment(), plus "image" key with loaded BGR image.
        """
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")

        result = self.segment(image, return_intermediates=return_intermediates)
        result["image"] = image
        return result

    def _fallback_parsing_only(
        self, parsing: np.ndarray, H: int, W: int
    ) -> Dict[str, np.ndarray]:
        """
        Fallback when landmark detection fails.
        Uses BiSeNet parsing + simple vertical/horizontal splits.
        """
        skin_mask = ((parsing == 1) * 255).astype(np.uint8)
        nose_mask = ((parsing == 10) * 255).astype(np.uint8)

        # Simple geometric splits on the skin mask
        # Find face bounding box from non-zero skin pixels
        ys, xs = np.where(skin_mask > 0)
        if len(xs) == 0:
            # No skin detected at all
            empty = np.zeros((H, W), dtype=np.uint8)
            return {
                "nose": nose_mask,
                "left_cheek": empty.copy(),
                "right_cheek": empty.copy(),
                "forehead": empty.copy(),
                "chin": empty.copy(),
            }

        face_cx = int(np.mean(xs))
        face_top = int(np.min(ys))
        face_bottom = int(np.max(ys))
        face_h = face_bottom - face_top

        # Rough vertical thirds
        upper_third = face_top + int(face_h * 0.33)
        lower_third = face_top + int(face_h * 0.66)

        forehead = skin_mask.copy()
        forehead[upper_third:, :] = 0

        middle_skin = skin_mask.copy()
        middle_skin[:upper_third, :] = 0
        middle_skin[lower_third:, :] = 0
        # Remove nose from cheeks
        middle_skin = cv2.bitwise_and(
            middle_skin, cv2.bitwise_not(nose_mask)
        )

        left_cheek = middle_skin.copy()
        left_cheek[:, :face_cx] = 0

        right_cheek = middle_skin.copy()
        right_cheek[:, face_cx:] = 0

        chin = skin_mask.copy()
        chin[:lower_third, :] = 0
        # Remove mouth/lips
        for lid in [11, 12, 13]:
            chin = cv2.bitwise_and(
                chin, cv2.bitwise_not(((parsing == lid) * 255).astype(np.uint8))
            )

        return {
            "nose": nose_mask,
            "left_cheek": left_cheek,
            "right_cheek": right_cheek,
            "forehead": forehead,
            "chin": chin,
        }
