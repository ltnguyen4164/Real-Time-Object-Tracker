# Real-Time-Object-Tracker
Real-time multi-object tracking system utilizing YOLOv8 for detection and a custom Kalman Filter implementation for state estimation and trajectory prediction

# Key Features
- State Estimation: Custom-built Kalman Filter to predict object position when the camera view is blocked or noisy.
- Intelligent Detection: Real-time inference using the YOLOv8 nano model for high-speed performance.
- Noise Reduction: Implemented MOG2 background subtraction and Gaussian blurring to clean the input stream before processing.
- Dynamic Visualization: Live feedback loop showing bounding boxes, class labels, and "estimated" center-points.

# Technical Implementation
- Prediction: The filter uses a constant velocity model (x=Fx+Bu) to guess where the object will be in the next frame.
- Update: When a new YOLO detection arrives, the system calculates the "Kalman Gain" to decide how much to trust the new data versus the old prediction.

# How to Run
### First install necessary libraries
```
pip install -r requirements.txt
```
### Run Python program
```
python main.py
```

# Why I Built This
I built this primarily to learn about Computer Vision topics like filtering images with morphology and Kalman Filters. While standard Computer Vision tools (like OpenCV's trackers) are effective, they often fail during occlusions (e.g., an object moving behind a wall) or when an object becomes stationary. To solve this, I implemented a custom Kalman Filter class to provide state estimation. This allows the system to maintain a "memory" of the object’s trajectory and predict its position even when visual detection is lost. Additionally, I integrated YOLOv8 to provide high-accuracy object classification, with a future roadmap to include automated alerting/pinging systems upon specific class detections (e.g., person detection).

# Acknowledgements
[Kalman Filter Tutorial](https://filterpy.readthedocs.io/en/latest/kalman/KalmanFilter.html)  
[Kalman Filter Code Structure](https://www.geeksforgeeks.org/python/kalman-filter-in-python/)
