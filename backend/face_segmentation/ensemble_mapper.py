"""
Ensemble Mapping Utility - V7 GOLD (Statistical Adaptive Gating Edition)
Fuses Roboflow API detections with Local Statistical Anomaly Detection.
Uses Patient-Specific Skin Baselines to eliminate consistent background noise.

Preserves per-detection class labels from typed models (e.g. Model B)
and maps them to clinical GAGS severity grades via the parent
LesionMapper.SEVERITY_MAP.

After NMS and statistical gating, a proximity-propagation pass assigns
typed labels to remaining generic ("Acne") detections when a nearby
Model B typed detection exists within ``PROXIMITY_THRESHOLD``.
"""
import cv2
import math
import numpy as np
from typing import Dict, List, Tuple, Optional
from .mapping import LesionMapper

# Generic labels that indicate the model does NOT distinguish types.
# These are treated as "untyped" and get the default severity grade.
_GENERIC_CLASS_NAMES = frozenset({'acne', 'acne_detected', 'lesion', ''})

# Maximum centre-to-centre distance (in normalised [0,1] coords) for
# proximity-based type propagation.  0.06 ≈ 6 % of image diagonal —
# roughly 1.5 cm at typical selfie distance.
PROXIMITY_THRESHOLD = float(0.06)


def _is_typed_label(class_name: str) -> bool:
    """Return True if *class_name* carries real type information."""
    return class_name.lower().strip() not in _GENERIC_CLASS_NAMES


class EnsembleLesionMapper(LesionMapper):
    """
    Advanced mapper using Statistical Adaptive Gating (SAG) and Anatomical Verification.
    Preserves per-detection acne type labels from the Roboflow response
    and resolves them to clinical GAGS severity grades.

    After gating, a proximity pass propagates typed labels from nearby
    Model B detections to generic "Acne" detections that survived NMS
    but had no direct type information.
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
        ensemble_assignments: Dict[str, List[Dict]] = {name: [] for name in self.region_names}
        ensemble_assignments['unassigned'] = []

        # 1. Calculate Patient-Specific Skin Baseline (SAG)
        skin_baseline_redness = 0.04  # Fallback
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

        def normalize_rf_preds(preds: List[Dict]) -> Tuple[List[List[float]], List[float], List[str]]:
            """Extract normalised boxes, confidence scores, and class labels."""
            boxes: List[List[float]] = []
            scores: List[float] = []
            class_names: List[str] = []
            for p in preds:
                x, y, w, h = p['x'], p['y'], p['width'], p['height']
                x1, y1 = (x - w / 2) / W, (y - h / 2) / H
                x2, y2 = (x + w / 2) / W, (y + h / 2) / H
                boxes.append([x1, y1, x2, y2])
                scores.append(p['confidence'])
                # Roboflow detection responses use the key "class"
                class_names.append(str(p.get('class', 'acne')))
            return boxes, scores, class_names

        # 2. Normalise all three prediction streams
        b_a1, s_a1, c_a1 = normalize_rf_preds(preds_a_640)
        b_a2, s_a2, c_a2 = normalize_rf_preds(preds_a_1280)
        b_b, s_b, c_b = normalize_rf_preds(preds_b)

        all_raw_boxes = b_a1 + b_a2 + b_b
        all_raw_scores = s_a1 + s_a2 + s_b
        all_raw_classes = c_a1 + c_a2 + c_b

        if not all_raw_scores:
            return ensemble_assignments

        # Build typed-source list from raw Model B predictions for
        # proximity propagation later.  Each entry is
        # (centre_x_norm, centre_y_norm, class_name).
        typed_sources: List[Tuple[float, float, str]] = []
        for box, cls in zip(b_b, c_b):
            if _is_typed_label(cls):
                cx_n = (box[0] + box[2]) / 2.0
                cy_n = (box[1] + box[3]) / 2.0
                typed_sources.append((cx_n, cy_n, cls))

        # 3. Sequential NMS — prefer higher-confidence detections but
        #    carry the *most specific* class label from any overlapping
        #    detection that gets suppressed.
        indices = np.argsort(all_raw_scores)[::-1]
        keep_indices: List[int] = []
        # Maps kept index -> best class label seen among its overlapping peers
        best_class_for_kept: Dict[int, str] = {}

        for i in indices:
            curr_box = all_raw_boxes[i]
            is_redundant = False
            for k_idx in keep_indices:
                if self._calculate_iou(curr_box, all_raw_boxes[k_idx]) > 0.35:
                    is_redundant = True
                    # If the suppressed detection has a typed label and the
                    # kept one does not, promote the typed label.
                    suppressed_cls = all_raw_classes[i]
                    if _is_typed_label(suppressed_cls) and not _is_typed_label(best_class_for_kept.get(k_idx, all_raw_classes[k_idx])):
                        best_class_for_kept[k_idx] = suppressed_cls
                    break
            if not is_redundant:
                keep_indices.append(i)
                best_class_for_kept[i] = all_raw_classes[i]

        # 4. Statistical Gating Loop
        for idx in keep_indices:
            b = all_raw_boxes[idx]
            s = all_raw_scores[idx]
            x1, y1, x2, y2 = int(b[0] * W), int(b[1] * H), int(b[2] * W), int(b[3] * H)
            w, h = (x2 - x1), (y2 - y1)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cx, cy = max(0, min(W - 1, cx)), max(0, min(H - 1, cy))

            # A. Anatomical check
            assigned_region = 'unassigned'
            for name, mask in self.region_masks.items():
                if mask[cy, cx] > 0:
                    assigned_region = name
                    break
            if assigned_region == 'unassigned':
                continue

            # B. Morphological check (Compactness)
            aspect_ratio = max(w, h) / (min(w, h) + 1e-6)
            if aspect_ratio > 2.2:
                continue  # Reject lines (hair/wrinkles)

            # C. Statistical Anomaly Detection (SAG)
            if image is not None:
                patch = image[max(0, y1):min(H, y2), max(0, x1):min(W, x2)]
                if patch.size > 0:
                    pr = patch[:, :, 2].astype(float)
                    pg = patch[:, :, 1].astype(float)
                    p_redness = np.mean((pr - pg) / (pr + pg + 1e-6))

                    # Threshold: Must be significantly redder than face average
                    # Using a Z-Score approach (Sigma gating)
                    z_score = (p_redness - skin_baseline_redness) / (skin_std_dev + 1e-6)

                    if z_score < 1.5:  # Must be > 1.5 Sigma outlier in redness
                        continue

            # Resolve class label: prefer the best (most specific) label
            # found during NMS overlap resolution.
            class_name = best_class_for_kept.get(idx, all_raw_classes[idx])
            severity_grade = self._get_severity_grade(class_name)
            type_source = 'direct' if _is_typed_label(class_name) else 'none'

            ensemble_assignments[assigned_region].append({
                'bbox': [x1, y1, x2, y2],
                'center': [cx, cy],
                'confidence': float(s),
                'reliability_score': round(float(s), 3),
                'class_name': class_name,
                'severity_grade': severity_grade,
                'type_source': type_source,
                'confidence_level': 'Statistically Verified',
                'votes': 1,
            })

        # 5. Proximity-based type propagation
        # For any surviving detection still carrying a generic label,
        # check if a raw Model B typed detection is nearby (centre-to-
        # centre Euclidean distance in normalised coords).  If so,
        # adopt its type and recalculate severity.
        if typed_sources:
            self._propagate_types_by_proximity(
                ensemble_assignments, typed_sources, H, W,
            )

        return ensemble_assignments

    # ------------------------------------------------------------------
    # Proximity propagation helper
    # ------------------------------------------------------------------

    def _propagate_types_by_proximity(
        self,
        assignments: Dict[str, List[Dict]],
        typed_sources: List[Tuple[float, float, str]],
        H: int,
        W: int,
    ) -> None:
        """Propagate typed labels to nearby generic detections in-place.

        Args:
            assignments: Region -> list-of-detection dicts (mutated).
            typed_sources: (cx_norm, cy_norm, class_name) from raw
                Model B typed predictions.
            H: Image height in pixels.
            W: Image width in pixels.
        """
        for region_name, detections in assignments.items():
            if region_name == 'unassigned':
                continue
            for det in detections:
                if det['type_source'] != 'none':
                    continue  # already has a real type
                # Normalised centre of this detection
                det_cx = det['center'][0] / W
                det_cy = det['center'][1] / H
                best_dist = float('inf')
                best_cls: Optional[str] = None
                for src_cx, src_cy, src_cls in typed_sources:
                    d = math.sqrt((det_cx - src_cx) ** 2 + (det_cy - src_cy) ** 2)
                    if d < best_dist:
                        best_dist = d
                        best_cls = src_cls
                if best_dist <= PROXIMITY_THRESHOLD and best_cls is not None:
                    det['class_name'] = best_cls
                    det['severity_grade'] = self._get_severity_grade(str(best_cls))
                    det['type_source'] = 'proximity'

    def ensemble_map_api(self, preds_a, preds_b, image_shape, image=None, weights=None, iou_thr=0.5):
        return self.ensemble_map_multi_scale(preds_a, [], preds_b, image_shape, image=image)

    @staticmethod
    def _calculate_iou(box1: List[float], box2: List[float]) -> float:
        from utils import calculate_iou
        return calculate_iou(box1, box2)
