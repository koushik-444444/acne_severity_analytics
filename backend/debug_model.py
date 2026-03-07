from ultralytics import YOLO
import cv2
import os

model = YOLO(r"E:\final_acne\bestk.pt")
img = cv2.imread(r"C:\Users\KOUSHIK\OneDrive\Desktop\acne_v7_api_only_gold_dist\final_verified_v7\10_anonymized.jpg")
results = model.predict(img, conf=0.01) # Ultra low confidence
for r in results:
    print(f"Detected {len(r.boxes)} boxes")
    for box in r.boxes:
        print(f"Box: {box.xyxy}, Conf: {box.conf}, Class: {box.cls}")
