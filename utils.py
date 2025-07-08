import cv2
import numpy as np
from scipy.spatial.distance import cosine

def extract_player_features(frame, bbox):
    x1, y1, x2, y2 = bbox
    patch = frame[y1:y2, x1:x2]
    patch = cv2.resize(patch, (64, 128))
    hist = cv2.calcHist([patch], [0, 1, 2], None, [8, 8, 8],
                        [0, 256, 0, 256, 0, 256])
    return cv2.normalize(hist, hist).flatten()

def match_players(frames1, frames2):
    mapping = {}
    for (f1, boxes1), (f2, boxes2) in zip(frames1, frames2):
        f1_feats = [extract_player_features(f1, b) for b in boxes1]
        f2_feats = [extract_player_features(f2, b) for b in boxes2]

        for i, feat1 in enumerate(f1_feats):
            best = -1
            best_sim = float("inf")
            for j, feat2 in enumerate(f2_feats):
                sim = cosine(feat1, feat2)
                if sim < best_sim:
                    best_sim = sim
                    best = j
            mapping[i] = best
    return mapping

def assign_ids(features, tracks, next_id, threshold=0.4):
    ids = []
    for feat in features:
        best_id = -1
        best_sim = float("inf")
        for tid, data in tracks.items():
            sim = cosine(feat, data['feature'])
            if sim < best_sim and sim < threshold:
                best_sim = sim
                best_id = tid
        if best_id == -1:
            ids.append(next_id)
            next_id += 1
        else:
            ids.append(best_id)
    return ids
