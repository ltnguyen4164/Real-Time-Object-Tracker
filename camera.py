import cv2
import numpy as np
import random
from ultralytics import YOLO

import filter

yolo = YOLO("yolov8n.pt")

# Generate unique colors for each class ID
def color(cls_num):
    random.seed(cls_num)
    return tuple(random.randint(0, 255) for _ in range(3))

dt = 1
filters = {}

cap = cv2.VideoCapture(0)

back = cv2.createBackgroundSubtractorMOG2(
    history=200,
    varThreshold=50,
    detectShadows=False
)

kernel = np.ones((3,3), np.uint8)

# Warm up background model
for _ in range(30):
    ret, frame = cap.read()
    if not ret:
        break
    back.apply(frame)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, None, fx=0.7, fy=0.7)
    results = list(yolo.track(frame, stream=True))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    fgmask = back.apply(blur)
    _, fgmask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)

    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=2)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8), iterations=2)

    # Temporal smoothing
    fgmask_f = fgmask.astype(np.float32)
    if 'accum' not in locals():
        accum = fgmask_f
    else:
        accum = cv2.addWeighted(accum, 0.7, fgmask_f, 0.3, 0)
    _, fgmask = cv2.threshold(accum.astype(np.uint8), 128, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    frameCopy = frame.copy()

    for result in results:
        class_names = result.names
        for box in result.boxes:
            if box.conf[0] < 0.4:
                continue
                
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            
            cls = int(box.cls[0])
            class_name = class_names[cls]
            conf = float(box.conf[0])

            track_id = int(box.id[0]) if box.id is not None else None
            if track_id is None:
                continue
            
            z = np.array([[cx],[cy]])
            if track_id not in filters:
                x0 = np.array([[cx],[cy],[0],[0]])
                filters[track_id] = filter.KalmanFilter(dt,x0)
            kf = filters[track_id]
            pred = kf.predict(np.zeros((4,1)))
            est = kf.update(z)

            px = int(est[0])
            py = int(est[1])

            colour = color(cls)

            cv2.rectangle(frameCopy, (x1, y1), (x2, y2), colour, 2)
            cv2.circle(frameCopy,(px,py),6,(0,255,0),-1)
            cv2.putText(frameCopy, f"{class_name} {conf:.2f}",
                        (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, colour, 2)

    active_ids = set()

    for result in results:
        for box in result.boxes:
            if box.id is not None:
                active_ids.add(int(box.id[0]))

    for track_id, kf in filters.items():
        if track_id not in active_ids:
            pred = kf.predict(np.zeros((4,1)))
            px = int(pred[0])
            py = int(pred[1])
            cv2.circle(frameCopy,(px,py),6,(0,0,255),-1)
    
    foreground = cv2.bitwise_and(frame, frame, mask=fgmask)
    stacked = np.hstack((frame, foreground, frameCopy))

    cv2.imshow("Live Detection", stacked)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()