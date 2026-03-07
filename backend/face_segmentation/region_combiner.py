"""
Two-Stage Region Combiner

Combines BiSeNet semantic segmentation (Stage 1) with landmark-based
geometric partitioning (Stage 2) to produce final pixel-accurate masks
for: nose, left_cheek, right_cheek, forehead, chin.

Strategy:
- Nose: Directly from BiSeNet label 10 (most accurate source)
- Skin mask: From BiSeNet label 1
- Forehead: Landmark geometry polygon INTERSECTED with skin mask
- Left cheek: Landmark geometry polygon INTERSECTED with skin mask
- Right cheek: Landmark geometry polygon INTERSECTED with skin mask
- Chin: Landmark geometry polygon INTERSECTED with skin mask

This intersection ensures:
1. Pixel accuracy from BiSeNet (no hair, eyes, eyebrows bleed into cheeks)
2. Geometric correctness from landmarks (separate L/R cheeks, forehead vs chin)
"""

from typing import Dict, Optional

import cv2
import numpy as np


class RegionCombiner:
    """
    Combines BiSeNet parsing masks with landmark-defined geometric regions
    to produce final face region masks.
    """

    # Morphological kernel for mask cleanup
    MORPH_KERNEL_SIZE = 5

    def __init__(self, morph_iterations: int = 1, smooth_edges: bool = False):
        """
        Args:
            morph_iterations: Number of morphological open/close iterations
                for mask cleanup. Set to 0 to disable.
            smooth_edges: If True, apply a slight Gaussian blur to mask edges
                (useful for soft blending).
        """
        self.morph_iterations = morph_iterations
        self.smooth_edges = smooth_edges
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.MORPH_KERNEL_SIZE, self.MORPH_KERNEL_SIZE),
        )

    def combine(
        self,
        parsing: np.ndarray,
        landmark_masks: Dict[str, np.ndarray],
        nose_from_bisenet: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Produce final region masks by combining parsing + landmark geometry.
        Uses a priority system to eliminate overlaps:
        Priority: Nose > Chin > Forehead > Cheeks > Jawline/Neck.

        Args:
            parsing: BiSeNet parsing map, shape (H, W), values 0-18.
            landmark_masks: Dict of landmark-based geometric masks.
            nose_from_bisenet: If True, use BiSeNet nose label directly.

        Returns:
            Dict mapping region name to binary mask (H, W).
        """
        # 1. Extract base masks from BiSeNet
        skin_mask = ((parsing == 1) * 255).astype(np.uint8)
        neck_label_mask = ((parsing == 14) * 255).astype(np.uint8)
        
        # AGGRESSIVE HOLE FILLING for skin mask
        skin_mask = cv2.morphologyEx(
            skin_mask, cv2.MORPH_CLOSE, 
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        )
        
        bisenet_nose = ((parsing == 10) * 255).astype(np.uint8)
        
        # Build mouth/lips mask for exclusion
        mouth_mask = np.zeros_like(skin_mask)
        for label_id in [11, 12, 13]:  # mouth, u_lip, l_lip
            mouth_mask = cv2.bitwise_or(
                mouth_mask, ((parsing == label_id) * 255).astype(np.uint8)
            )

        # 2. Sequential Extraction (High Priority to Low Priority)
        assigned_mask = np.zeros_like(skin_mask)
        result = {}

        def get_geo(name):
            m = landmark_masks.get(name)
            return m if m is not None else np.zeros_like(skin_mask)

        # --- Nose (Priority 1) ---
        if nose_from_bisenet:
            nose_final = self._clean_mask(bisenet_nose)
        else:
            nose_geo = get_geo("nose")
            nose_base = cv2.bitwise_or(skin_mask, bisenet_nose)
            nose_final = self._clean_mask(cv2.bitwise_and(nose_geo, nose_base))
        
        result["nose"] = nose_final
        assigned_mask = cv2.bitwise_or(assigned_mask, nose_final)

        # --- Chin (Priority 2) ---
        chin_geo = get_geo("chin")
        chin_combined = cv2.bitwise_and(chin_geo, skin_mask)
        chin_combined = cv2.bitwise_and(chin_combined, cv2.bitwise_not(mouth_mask))
        chin_combined = cv2.bitwise_and(chin_combined, cv2.bitwise_not(assigned_mask))
        
        result["chin"] = self._clean_mask(chin_combined)
        assigned_mask = cv2.bitwise_or(assigned_mask, result["chin"])

        # --- Forehead (Priority 3) ---
        forehead_geo = get_geo("forehead")
        forehead_combined = cv2.bitwise_and(forehead_geo, skin_mask)
        forehead_combined = cv2.bitwise_and(forehead_combined, cv2.bitwise_not(assigned_mask))
        
        result["forehead"] = self._clean_mask(forehead_combined)
        assigned_mask = cv2.bitwise_or(assigned_mask, result["forehead"])

        # --- Cheeks (Priority 4) with Strict Lateral Split ---
        H, W = skin_mask.shape
        # Create a hard vertical split at the nose center to prevent L/R overlap
        # We find the average X of the center_line landmarks if available
        # landmark_masks handles the polygons, but we'll enforce the split here.
        
        for side in ["left_cheek", "right_cheek"]:
            cheek_geo = get_geo(side)
            cheek_combined = cv2.bitwise_and(cheek_geo, skin_mask)
            cheek_combined = cv2.bitwise_and(cheek_combined, cv2.bitwise_not(mouth_mask))
            cheek_combined = cv2.bitwise_and(cheek_combined, cv2.bitwise_not(assigned_mask))
            
            # Enforce hard lateral split to avoid double counting at the boundary
            # If any pixels overlap between L and R after individual processing, 
            # they were already handled by assigned_mask? No, cheeks are lower priority.
            # Let's add them to assigned_mask one by one to prevent their OWN overlap.
            result[side] = self._clean_mask(cheek_combined)
            assigned_mask = cv2.bitwise_or(assigned_mask, result[side])

        # --- Jawline / Upper Neck (Priority 5) ---
        # Intersect landmark neck zone with BiSeNet neck label
        neck_geo = get_geo("jawline_neck")
        neck_combined = cv2.bitwise_and(neck_geo, neck_label_mask)
        # Also include bits of skin label that are in the neck zone but not assigned to face
        neck_extra = cv2.bitwise_and(neck_geo, skin_mask)
        neck_extra = cv2.bitwise_and(neck_extra, cv2.bitwise_not(assigned_mask))
        neck_final = cv2.bitwise_or(neck_combined, neck_extra)
        # Exclude everything else
        neck_final = cv2.bitwise_and(neck_final, cv2.bitwise_not(assigned_mask))
        
        result["jawline_neck"] = self._clean_mask(neck_final)

        # Apply smoothing if requested
        if self.smooth_edges:
            for name in result:
                blurred = cv2.GaussianBlur(result[name], (11, 11), 0)
                result[name] = blurred

        return result

    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        """Apply morphological operations to clean up mask edges."""
        if self.morph_iterations <= 0:
            return mask

        # Close small holes
        cleaned = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, self.kernel,
            iterations=self.morph_iterations,
        )
        # Remove small noise
        cleaned = cv2.morphologyEx(
            cleaned, cv2.MORPH_OPEN, self.kernel,
            iterations=self.morph_iterations,
        )
        return cleaned

    @staticmethod
    def compute_coverage(masks: Dict[str, np.ndarray]) -> Dict[str, int]:
        """Compute pixel count for each region mask."""
        return {name: int(np.sum(mask > 0)) for name, mask in masks.items()}

    @staticmethod
    def check_overlap(masks: Dict[str, np.ndarray]) -> Dict[str, int]:
        """Check for overlapping pixels between region masks."""
        region_names = list(masks.keys())
        overlaps = {}
        for i in range(len(region_names)):
            for j in range(i + 1, len(region_names)):
                name_i, name_j = region_names[i], region_names[j]
                overlap = cv2.bitwise_and(masks[name_i], masks[name_j])
                overlap_count = int(np.sum(overlap > 0))
                if overlap_count > 0:
                    overlaps[f"{name_i} & {name_j}"] = overlap_count
        return overlaps
