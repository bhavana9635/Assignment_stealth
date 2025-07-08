from ultralytics import YOLO
import cv2
import numpy as np
from utils import extract_player_features, assign_ids

model = YOLO("best.pt")

cap = cv2.VideoCapture("15sec_input_720p.mp4")
tracks = {}
next_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]
    players = [r for r in results.boxes.data.cpu().numpy() if int(r[5]) == 0]
    bboxes = [(int(x1), int(y1), int(x2), int(y2)) for x1, y1, x2, y2, _, _ in players]

    features = [extract_player_features(frame, box) for box in bboxes]
    tracked_ids = assign_ids(features, tracks, next_id)

    # Update state
    for tid, box in zip(tracked_ids, bboxes):
        tracks[tid] = {'bbox': box, 'feature': extract_player_features(frame, box)}
    next_id = max(tracked_ids + [next_id]) + 1

    # Visualization (Optional)
    for tid, box in zip(tracked_ids, bboxes):
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {tid}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.imshow("Re-ID", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
