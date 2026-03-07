"""
Mapping Utility - Assigns YOLO detections to facial regions using BiSeNet/MediaPipe masks.
Includes Clinical LPI (Lesion-to-Pixel Index) and GAGS (Global Acne Grading System).
"""
import numpy as np
from typing import Dict, List, Tuple, Optional

class LesionMapper:
    """
    Maps acne detections (bounding boxes) to specific facial regions.
    Calculates clinical density and severity scores.
    """

    # GAGS Regional Weights
    REGION_WEIGHTS = {
        "forehead": 2,
        "right_cheek": 2,
        "left_cheek": 2,
        "nose": 1,
        "chin": 1,
        "jawline_neck": 1
    }

    # Clinical Severity Grade Mapping (typical for Roboflow acne models)
    # Map class ID or name to GAGS Grade (1-4)
    # 1: Comedone, 2: Papule, 3: Pustule, 4: Nodule/Cyst
    SEVERITY_MAP = {
        "blackhead": 1,
        "whitehead": 1,
        "comedone": 1,
        "papule": 2,
        "pustule": 3,
        "nodule": 4,
        "cyst": 4,
        "cystic": 4
    }

    def __init__(self, region_masks: Dict[str, np.ndarray]):
        """
        Args:
            region_masks: Dict mapping region name to binary numpy mask (H, W).
        """
        self.region_masks = region_masks
        self.region_names = list(region_masks.keys())

    def map_lesions(self, boxes: np.ndarray, class_names: Optional[List[str]] = None) -> Dict[str, List[Dict]]:
        """
        Assign each box to a facial region.
        
        Args:
            boxes: (N, 6) array of [x1, y1, x2, y2, conf, cls_id]
            class_names: List of class strings corresponding to cls_id.
                  
        Returns:
            Dict mapping region name to list of detection info.
        """
        assignments = {name: [] for name in self.region_names}
        assignments["unassigned"] = []

        for box in boxes:
            x1, y1, x2, y2 = box[:4].astype(int)
            conf = float(box[4])
            cls_id = int(box[5])
            
            # Determine class name and clinical severity grade
            class_name = class_names[cls_id] if class_names and cls_id < len(class_names) else "acne"
            severity_grade = self._get_severity_grade(class_name)
            
            # Calculate center point
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            # 1. Point-in-mask check
            assigned_region = "unassigned"
            H, W = next(iter(self.region_masks.values())).shape[:2]
            cx, cy = max(0, min(W - 1, cx)), max(0, min(H - 1, cy))

            for name, mask in self.region_masks.items():
                if mask[cy, cx] > 0:
                    assigned_region = name
                    break
            
            # 2. Majority Area check (Fallback)
            if assigned_region == "unassigned":
                max_area = 0
                for name, mask in self.region_masks.items():
                    region_crop = mask[max(0,y1):min(H,y2), max(0,x1):min(W,x2)]
                    area = np.sum(region_crop > 0)
                    if area > max_area:
                        max_area = area
                        assigned_region = name

            lesion_info = {
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "center": [int(cx), int(cy)],
                "confidence": conf,
                "class_id": cls_id,
                "class_name": class_name,
                "severity_grade": severity_grade
            }
            assignments[assigned_region].append(lesion_info)

        return assignments

    def _get_severity_grade(self, class_name: str) -> int:
        """Map class name to clinical grade (1-4). Default 2."""
        name_lower = class_name.lower()
        for key, grade in self.SEVERITY_MAP.items():
            if key in name_lower:
                return grade
        return 2 # Moderate default

    def get_clinical_report(self, assignments: Dict[str, List[Dict]]) -> Dict:
        """
        Calculate LPI density and GAGS scoring.
        """
        report = {
            "regions": {},
            "total_lesions": 0,
            "gags_total_score": 0,
            "clinical_severity": "None",
            "symmetry_delta": 0.0
        }

        total_lesions = 0
        for name, items in assignments.items():
            if name == "unassigned": continue
            
            count = len(items)
            total_lesions += count
            
            # Regional Area
            area = np.sum(self.region_masks[name] > 0)
            
            # LPI (Lesion-to-Pixel Index) per Million Pixels
            lpi = (count / area * 1_000_000) if area > 0 else 0
            
            # Regional GAGS Score = Weight x Max Grade in region
            weight = self.REGION_WEIGHTS.get(name, 1)
            max_grade = max([it["severity_grade"] for it in items]) if items else 0
            regional_score = weight * max_grade
            
            report["regions"][name] = {
                "count": count,
                "lpi": round(lpi, 2),
                "area_px": int(area),
                "gags_score": regional_score
            }
            report["gags_total_score"] += regional_score

        report["total_lesions"] = total_lesions
        
        # Clinical Severity Mapping
        score = report["gags_total_score"]
        if score == 0: report["clinical_severity"] = "None"
        elif score <= 18: report["clinical_severity"] = "Mild"
        elif score <= 30: report["clinical_severity"] = "Moderate"
        elif score <= 38: report["clinical_severity"] = "Severe"
        else: report["clinical_severity"] = "Very Severe / Cystic"

        # Symmetry Analysis (LPI difference between cheeks)
        l_lpi = report["regions"].get("left_cheek", {}).get("lpi", 0)
        r_lpi = report["regions"].get("right_cheek", {}).get("lpi", 0)
        if (l_lpi + r_lpi) > 0:
            report["symmetry_delta"] = round(abs(l_lpi - r_lpi) / max(l_lpi, r_lpi) * 100, 1)

        return report

    @staticmethod
    def get_summary_report(assignments: Dict[str, List[Dict]]) -> Dict:
        """Legacy summary for backward compatibility."""
        return {
            "counts": {name: len(items) for name, items in assignments.items()},
            "total_lesions": sum(len(items) for items in assignments.values())
        }
