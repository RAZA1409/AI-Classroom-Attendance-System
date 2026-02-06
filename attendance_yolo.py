import time
import os
from datetime import datetime
from ultralytics import YOLO
import cv2

# ==============================
# Load YOLO model
# ==============================
model = YOLO("yolov8n.pt")

# ==============================
# Stable ID mapping
# ==============================
id_map = {}          # YOLO_ID -> Stable Person_ID
next_person_id = 1

# ==============================
# FPS variables
# ==============================
prev_time = 0

# ==============================
# Attendance logic variables
# ==============================
first_seen = {}      # person_id -> first detection timestamp
marked_ids = set()   # already marked attendance
status_text = {}     # person_id -> status string
MIN_TIME = 5         # seconds required to mark attendance
COOLDOWN_TIME = 600  # seconds (10 minutes)
last_marked_time = {}  # person_id -> last marked timestamp


# ==============================
# Open webcam
# ==============================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Camera not accessible")
    exit()

# ==============================
# Create CSV file with header
# ==============================
if not os.path.exists("attendance.csv"):
    with open("attendance.csv", "w") as f:
        f.write("Date,Person_ID,Time,Duration,Status\n")

print("🚀 AI Classroom Attendance System Started")
print("Press 'q' to quit")

# ==============================
# Main loop
# ==============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # FPS calculation
    curr_time = time.time()
    fps = int(1 / (curr_time - prev_time)) if prev_time != 0 else 0
    prev_time = curr_time

    # Run YOLO tracking
    results = model.track(frame, classes=[0], conf=0.5, persist=True)

    annotated = frame.copy()
    current_time = time.time()

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            yolo_id = int(box.id[0]) if box.id is not None else None
            if yolo_id is None:
                continue

            # Assign stable person ID
            if yolo_id not in id_map:
                id_map[yolo_id] = next_person_id
                next_person_id += 1

            person_id = id_map[yolo_id]

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # First detection time
            if person_id not in first_seen:
                first_seen[person_id] = current_time

            duration = current_time - first_seen[person_id]

            # Status logic
            if duration < MIN_TIME:
                status_text[person_id] = f"Detecting {int(duration)}s"
                color = (0, 255, 255)  # Yellow
            else:
                status_text[person_id] = "Attendance Marked"
                color = (0, 255, 0)    # Green

            label = f"ID {person_id} | {status_text[person_id]}"

            # Draw bounding box and label
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

            # Mark attendance only once
            # if duration >= MIN_TIME and person_id not in marked_ids:
            can_mark = False

            if person_id not in last_marked_time:
                can_mark = True
            elif current_time - last_marked_time[person_id] >= COOLDOWN_TIME:
                can_mark = True

            if duration >= MIN_TIME and can_mark:
                marked_ids.add(person_id)
                last_marked_time[person_id] = current_time
                now = datetime.now()
                with open("attendance.csv", "a") as f:
                    f.write(
                        f"{now.strftime('%Y-%m-%d')},"
                        f"{person_id},"
                        f"{now.strftime('%H:%M:%S')},"
                        f"{int(duration)}s,"
                        f"Present\n"
                    )

                print(f"✅ Attendance marked for ID {person_id}")

    # Show FPS
    cv2.putText(
        annotated,
        f"FPS: {fps}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("AI Attendance System", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==============================
# Cleanup
# ==============================
cap.release()
cv2.destroyAllWindows()
print("🛑 System stopped")








