import cv2
import numpy as np

from vision.tracker import VideoTracker
from vision.audio import AlarmSystem

# Initialize
cap = cv2.VideoCapture(0)
tracker = VideoTracker()
alarm = AlarmSystem("audio/faaah.mp3")
back = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=50)

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.resize(frame, None, fx=0.7, fy=0.7)

    # Motion Check
    fgmask = back.apply(cv2.GaussianBlur(frame, (5,5), 0))
    has_motion = cv2.countNonZero(fgmask) > 500

    # The "Clean" Tracking Call
    tracked_objects, active_ids = tracker.update(frame, has_motion)

    for tid in active_ids:
        # Check if this specific tracker is a human
        if tracker.get_class(tid) == "person":
            alarm.trigger()

    # Visualization
    for tid, kf in tracked_objects.items():
        px, py = int(kf.x[0]), int(kf.x[1])
        color = (0, 255, 0) if tid in active_ids else (0, 0, 255)
        cv2.circle(frame, (px, py), 5, color, -1)
        cv2.putText(frame, f"ID:{tid}", (px, py-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    cv2.imshow("Clean Main Loop", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()