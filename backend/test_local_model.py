import cv2
from ultralytics import YOLO
import sys
import os

MODEL_PATH = r"E:\final_acne\bestk.pt"

def test_model():
    print(f"Loading local model: {MODEL_PATH}")
    try:
        model = YOLO(MODEL_PATH)
        print("Model loaded successfully.")
        
        # Check classes
        print(f"Classes: {model.names}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_model()
