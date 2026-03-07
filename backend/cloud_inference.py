"""
Cloud Inference Engine - V7 Multi-Scale Parallel Version
Handles Roboflow API communication with automated resolution scaling
and concurrent request handling for minimum latency.
"""
import os
import cv2
import time
import json
import concurrent.futures
from typing import List, Dict, Optional
from roboflow import Roboflow
from usage_tracker import log_api_call

class CloudInferenceEngine:
    def __init__(self, api_key: str):
        self.rf = Roboflow(api_key=api_key)
        self.max_api_dim = int(os.getenv("MAX_API_DIM", 2048))

    def fetch_multi_scale_consensus(
        self, 
        image: cv2.Mat, 
        model_a_id: str, 
        model_b_id: str
    ) -> Dict[str, List[Dict]]:
        """
        Executes the 'Triple-Look Strategy' in parallel.
        Returns detections for:
        1. Model A @ 640px
        2. Model A @ 1280px
        3. Model B @ Native/2048px
        """
        H, W = image.shape[:2]
        
        # Define tasks for the thread pool
        tasks = [
            (model_a_id, 640),
            (model_a_id, 1280),
            (model_b_id, self.max_api_dim)
        ]

        results = {}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_to_task = {
                executor.submit(self._fetch_single_scale, image, m_id, dim): f"{m_id}_{dim}"
                for m_id, dim in tasks
            }
            
            for future in concurrent.futures.as_completed(future_to_task):
                task_name = future_to_task[future]
                try:
                    results[task_name] = future.result()
                except Exception as e:
                    print(f"[Cloud Engine Error] {task_name}: {e}")
                    results[task_name] = []

        return {
            "preds_a_640": results.get(f"{model_a_id}_640", []),
            "preds_a_1280": results.get(f"{model_a_id}_1280", []),
            "preds_b": results.get(f"{model_b_id}_{self.max_api_dim}", [])
        }

    def _fetch_single_scale(self, image: cv2.Mat, model_id: str, target_dim: int) -> List[Dict]:
        """Internal helper for single API call with scaling."""
        parts = model_id.split("/")
        ws = parts[0] if len(parts) == 3 else "runner-e0dmy"
        proj = parts[1] if len(parts) == 3 else parts[0]
        ver = parts[2] if len(parts) == 3 else parts[1]
        
        H, W = image.shape[:2]
        model = self.rf.workspace(ws).project(proj).version(int(ver)).model
        
        # Scaling logic
        if max(H, W) > target_dim:
            scale = target_dim / max(H, W)
            temp = cv2.resize(image, (int(W*scale), int(H*scale)))
            temp_path = f"temp_api_{proj}_{target_dim}_{time.time()}.jpg"
            cv2.imwrite(temp_path, temp)
            
            try:
                res = model.predict(temp_path, confidence=10).json()
                preds = res.get("predictions", [])
                # Re-scale coordinates back to original image size
                for p in preds:
                    p["x"] /= scale; p["y"] /= scale
                    p["width"] /= scale; p["height"] /= scale
                log_api_call(model_id, "success")
                return preds
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        else:
            # Native resolution
            temp_path = f"temp_native_{proj}_{time.time()}.jpg"
            cv2.imwrite(temp_path, image)
            try:
                res = model.predict(temp_path, confidence=35).json()
                log_api_call(model_id, "success")
                return res.get("predictions", [])
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
