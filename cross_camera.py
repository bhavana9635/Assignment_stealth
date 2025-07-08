from ultralytics import YOLO
import cv2
import numpy as np
from utils import extract_player_features, match_players

model = YOLO("best.pt")

def detect_players(video_path):
    cap = cv2.VideoCapture(video_path)
    detections = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        players = [r for r in results.boxes.data.cpu().numpy() if int(r[5]) == 0]  # class 0 = player
        dets = [(int(x1), int(y1), int(x2), int(y2)) for x1, y1, x2, y2, _, _ in players]
        detections.append((frame, dets))

    cap.release()
    return detections

# Load detections
broadcast = detect_players("broadcast.mp4")
tacticam = detect_players("tacticam.mp4")

# Match using frame-by-frame player features
id_mapping = match_players(broadcast, tacticam)
print("Final Mapping Broadcast → Tacticam:", id_mapping)
