import cv2
import numpy as np
from ultralytics import YOLO

import vision.filter as filter

class VideoTracker:
    def __init__(self, model_path="models/yolov8n.pt", skip_frames=2, max_lost=30):
        self.model = YOLO(model_path)
        self.skip_frames = skip_frames
        self.max_lost = max_lost
        self.filters = {}
        self.tracker_age = {}
        self.frame_count = 0
        self.dt = 1
        self.object_classes = {}

    def update(self, frame, has_motion):
        self.frame_count += 1
        active_ids = set()

        # 1. Predict state for all existing trackers
        for kf in self.filters.values():
            kf.predict()

        # 2. Update with YOLO if conditions are met
        if has_motion and (self.frame_count % (self.skip_frames + 1) == 0):
            results = self.model.track(frame, persist=True, stream=True)
            for result in results:
                names = result.names
                
                for box in result.boxes:
                    if box.id is None or box.conf[0] < 0.4:
                        continue
                    
                    tid = int(box.id[0])
                    active_ids.add(tid)
                    self.tracker_age[tid] = 0
                    
                    class_idx = int(box.cls[0])
                    class_name = names[class_idx]
                    self.object_classes[tid] = class_name
                    
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    z = np.array([[(x1 + x2) // 2], [(y1 + y2) // 2]])

                    if tid not in self.filters:
                        x0 = np.array([[z[0,0]], [z[1,0]], [0], [0]])
                        self.filters[tid] = filter.KalmanFilter(self.dt, x0)
                    
                    self.filters[tid].update(z)

        # 3. Handle lost tracks
        self._manage_age(active_ids)
        return self.filters, active_ids

    def _manage_age(self, active_ids):
        for tid in list(self.filters.keys()):
            if tid not in active_ids:
                self.tracker_age[tid] = self.tracker_age.get(tid, 0) + 1
                if self.tracker_age[tid] > self.max_lost:
                    del self.filters[tid]
                    del self.tracker_age[tid]

                    if tid in self.object_classes:
                        del self.object_classes[tid]

    def get_class(self, tid):
        return self.object_classes.get(tid, "unknown")