# Player Re-Identification and Cross-Camera Mapping

## Overview
This project contains two main tasks:
1.Cross-Camera Player Mapping
2.Re-Identification in a Single Feed

A YOLOv11-based object detection model is used to detect players and perform consistent identity assignment across frames or video feeds.

## Setup

### Dependencies
```bash
pip install ultralytics opencv-python numpy scipy
```

### Project Structure
```
project/
├── broadcast.mp4              
├── tacticam.mp4               
├── 15sec_input_720p.mp4       
├── best.pt              
├── cross_camera.py    
├── reid_single_feed.py
├── utils.py                   
├── README.md                                 
```

## Running the Code

### Cross Camera Mapping
```bash
python cross_camera.py
```

### Re-ID on Single Feed
```bash
python reid_single_feed.py
```

## Notes
- Place your videos and YOLO model in the project directory.
- Adjust thresholds in `utils.py` if needed for feature matching.
