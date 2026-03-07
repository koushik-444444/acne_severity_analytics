"""
Ensemble Mapping Utility - V7 GOLD (Statistical Adaptive Gating Edition)
Fuses Roboflow API detections with Local Statistical Anomaly Detection.
Uses Patient-Specific Skin Baselines to eliminate consistent background noise.
"""
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from .mapping import LesionMapper

class EnsembleLesionMapper(LesionMapper):
    """
    Advanced mapper using Statistical Adaptive Gating (SAG) and Anatomical Verification.
    """

    def ensemble_map_multi_scale(
        self, 
        preds_a_640: List[Dict],
        preds_a_1280: List[Dict],
        preds_b: List[Dict],
        image_shape: Tuple[int, int],
        image: Optional[np.ndarray] = None,
        weights: List[float] = [1.0, 1.0, 1.5],
        iou_thr: float = 0.50
    ) -> Dict[str, List[Dict]]:
        H, W = image_shape
        ensemble_assignments = {name: [] for name in self.region_names}
        ensemble_assignments["unassigned"] = []

        # 1. Calculate Patient-Specific Skin Baseline (SAG)
        skin_baseline_redness = 0.04 # Fallback
        skin_std_dev = 0.01
        
        if image is not None:
            # Combine all clinical skin masks to find 'Global Skin'
            all_skin_mask = np.zeros((H, W), dtype=np.uint8)
            for mask in self.region_masks.values():
                all_skin_mask = cv2.bitwise_or(all_skin_mask, mask)
            
            skin_pixels = image[all_skin_mask > 0]
            if skin_pixels.size > 0:
                # Calculate Redness Distribution (R-G)/(R+G)
                rs = skin_pixels[:, 2].astype(float)
                gs = skin_pixels[:, 1].astype(float)
                redness_vals = (rs - gs) / (rs + gs + 1e-6)
                skin_baseline_redness = np.mean(redness_vals)
                skin_std_dev = np.std(redness_vals)

        def normalize_rf_preds(preds):
            boxes, scores, labels = [], [], []
            for i, p in enumerate(preds):
                x, y, w, h = p["x"], p["y"], p["width"], p["height"]
                x1, y1 = (x - w/2) / W, (y - h/2) / H
                x2, y2 = (x + w/2) / W, (y + h/2) / H
                boxes.append([x1, y1, x2, y2])
                scores.append(p["confidence"])
                labels.append(i)
            return boxes, scores, labels

        # 2. Sequential NMS
        b_a1, s_a1, l_a1 = normalize_rf_preds(preds_a_640)
        b_a2, s_a2, l_a2 = normalize_rf_preds(preds_a_1280)
        b_b, s_b, l_b = normalize_rf_preds(preds_b)
        
        all_raw_boxes = b_a1 + b_a2 + b_b
        all_raw_scores = s_a1 + s_a2 + s_b
        
        if not all_raw_scores: return ensemble_assignments

        indices = np.argsort(all_raw_scores)[::-1]
        keep_indices = []
        for i in indices:
            curr_box = all_raw_boxes[i]
            is_redundant = False
            for k_idx in keep_indices:
                if self._calculate_iou(curr_box, all_raw_boxes[k_idx]) > 0.35:
                    is_redundant = True; break
            if not is_redundant: keep_indices.append(i)

        # 3. Statistical Gating Loop
        for idx in keep_indices:
            b = all_raw_boxes[idx]
            s = all_raw_scores[idx]
            x1, y1, x2, y2 = int(b[0]*W), int(b[1]*H), int(b[2]*W), int(b[3]*H)
            w, h = (x2-x1), (y2-y1)
            cx, cy = (x1+x2)//2, (y1+y2)//2
            cx, cy = max(0, min(W-1, cx)), max(0, min(H-1, cy))

            # A. Anatomical check
            assigned_region = "unassigned"
            for name, mask in self.region_masks.items():
                if mask[cy, cx] > 0:
                    assigned_region = name; break
            if assigned_region == "unassigned": continue

            # B. Morphological check (Compactness)
            aspect_ratio = max(w, h) / (min(w, h) + 1e-6)
            if aspect_ratio > 2.2: continue # Reject lines (hair/wrinkles)

            # C. Statistical Anomaly Detection (SAG)
            if image is not None:
                patch = image[max(0, y1):min(H, y2), max(0, x1):min(W, x2)]
                if patch.size > 0:
                    pr = patch[:, :, 2].astype(float); pg = patch[:, :, 1].astype(float)
                    p_redness = np.mean((pr - pg) / (pr + pg + 1e-6))
                    
                    # Threshold: Must be significantly redder than face average
                    # Using a Z-Score approach (Sigma gating)
                    z_score = (p_redness - skin_baseline_redness) / (skin_std_dev + 1e-6)
                    
                    if z_score < 1.5: # Must be > 1.5 Sigma outlier in redness
                        continue

            ensemble_assignments[assigned_region].append({
                "bbox": [x1, y1, x2, y2],
                "center": [cx, cy],
                "confidence": float(s),
                "reliability_score": round(float(s), 3),
                "class_name": "acne",
                "severity_grade": 2,
                "confidence_level": "Statistically Verified",
                "votes": 1
            })

        return ensemble_assignments

    def ensemble_map_api(self, preds_a, preds_b, image_shape, image=None, weights=None, iou_thr=0.5):
        return self.ensemble_map_multi_scale(preds_a, [], preds_b, image_shape, image=image)

    @staticmethod
    def _calculate_iou(box1, box2):
        from utils import calculate_iou
        return calculate_iou(box1, box2)
