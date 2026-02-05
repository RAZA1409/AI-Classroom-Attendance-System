import time
from datetime import datetime
from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolov8n.pt")

# Attendance logic
first_seen = {}      # person_id -> first detection time
marked_ids = set()   # already marked attendance
MIN_TIME = 5         # seconds

# Open camera
cap = cv2.VideoCapture(0)

# Create CSV header if not exists
with open("attendance.csv", "a") as f:
    f.write("Date,Person_ID,Time\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, classes=[0], conf=0.5, persist=True)

    annotated = frame.copy()

    for r in results:
        if r.boxes is None:
            continue

        current_time = time.time()

        for box in r.boxes:
            person_id = int(box.id[0]) if box.id is not None else None
            if person_id is None:
                continue

            # First time detection
            if person_id not in first_seen:
                first_seen[person_id] = current_time

            duration = current_time - first_seen[person_id]

            # Mark attendance after MIN_TIME
            if duration >= MIN_TIME and person_id not in marked_ids:
                marked_ids.add(person_id)

                now = datetime.now()
                with open("attendance.csv", "a") as f:
                    f.write(
                        f"{now.strftime('%Y-%m-%d')},"
                        f"{person_id},"
                        f"{now.strftime('%H:%M:%S')}\n"
                    )

                print(f"✅ Attendance marked for ID {person_id}")

        annotated = r.plot()

    cv2.imshow("AI Attendance System", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

