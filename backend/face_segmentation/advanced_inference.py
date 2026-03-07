"""
V6 Advanced Inference Engine
Integrates SAHI (Slicing Aided Hyper Inference) for small object detection
and WBF (Weighted Box Fusion) for high-quality model ensembling.
"""
import os
import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from ensemble_boxes import weighted_boxes_fusion

try:
    from sahi.predict import get_sliced_prediction
    from sahi.models.ultralytics import UltralyticsDetectionModel
    _SAHI_AVAILABLE = True
except ImportError:
    _SAHI_AVAILABLE = False

class AdvancedInferenceEngine:
    """
    Handles high-quality inference using SAHI and WBF.
    """

    def __init__(self, model_paths: List[str], device: str = "cpu"):
        """
        Args:
            model_paths: List of local .pt model paths.
            device: 'cuda' or 'cpu'.
        """
        self.model_paths = model_paths
        self.device = device
        self._models = []
        self._sahi_models = []
        
        # Load models using Ultralytics for WBF/Standard
        from ultralytics import YOLO
        for path in model_paths:
            if os.path.exists(path):
                self._models.append(YOLO(path).to(device))
                if _SAHI_AVAILABLE:
                    self._sahi_models.append(UltralyticsDetectionModel(
                        model_path=path,
                        device=device,
                        confidence_threshold=0.1
                    ))
            else:
                print(f"[Warning] Model path not found: {path}")

    def predict_with_sahi(
        self, 
        image: np.ndarray, 
        slice_size: int = 640, 
        overlap_ratio: float = 0.2
    ) -> List[Dict]:
        """
        Run Sliced Inference on all loaded models.
        """
        if not self._sahi_models:
            return []

        all_model_preds = []
        for i, sahi_model in enumerate(self._sahi_models):
            result = get_sliced_prediction(
                image,
                sahi_model,
                slice_height=slice_size,
                slice_width=slice_size,
                overlap_height_ratio=overlap_ratio,
                overlap_width_ratio=overlap_ratio,
                verbose=0
            )
            
            # Convert to normalized xyxy for WBF
            boxes, scores, labels = [], [], []
            img_w, img_h = result.image_width, result.image_height
            for obj in result.object_prediction_list:
                bbox = obj.bbox.to_xyxy()
                boxes.append([bbox[0]/img_w, bbox[1]/img_h, bbox[2]/img_w, bbox[3]/img_h])
                scores.append(float(obj.score.value))
                labels.append(int(obj.category.id))
            
            # Get category names from the underlying model
            # Ultralytics model has .names attribute
            model_names = self._models[i].names
            
            all_model_preds.append({
                "boxes": boxes,
                "scores": scores,
                "labels": labels,
                "names": model_names
            })
        
        return self._fuse_results(all_model_preds)

    def _fuse_results(self, all_preds: List[Dict], iou_thr: float = 0.5) -> List[Dict]:
        """
        Fuse results from multiple models using WBF.
        """
        if not all_preds:
            return []
            
        boxes_list = [p["boxes"] for p in all_preds]
        scores_list = [p["scores"] for p in all_preds]
        labels_list = [p["labels"] for p in all_preds]
        
        # WBF requires lists of lists
        f_boxes, f_scores, f_labels = weighted_boxes_fusion(
            boxes_list, scores_list, labels_list, 
            iou_thr=iou_thr, skip_box_thr=0.1
        )
        
        # Convert back to pixel coordinates
        # (Assuming all models saw the same image size)
        # We'll use names from the first model
        names = all_preds[0]["names"]
        
        final_detections = []
        for b, s, l in zip(f_boxes, f_scores, f_labels):
            final_detections.append({
                "bbox_norm": b.tolist(), # [x1, y1, x2, y2]
                "confidence": float(s),
                "class_id": int(l),
                "class_name": names[int(l)] if int(l) < len(names) else "acne"
            })
            
        return final_detections
