import time
import os
from datetime import datetime
from ultralytics import YOLO
from face_database import load_database, recognize_face
import cv2
from collections import defaultdict

# ==============================
# Load YOLO model
# ==============================
model = YOLO("yolov8n.pt")

# ==============================
# Load face database (cached embeddings)
# ==============================
face_db = load_database()

# ==============================
# Tracking structures
# ==============================
id_map = {}
next_person_id = 1

track_meta = {}
person_names = {}

name_history = defaultdict(list)

TRACK_TIMEOUT = 2.0

# ==============================
# Attendance variables
# ==============================
marked_ids = set()
status_text = {}

MIN_TIME = 5
MIN_FRAMES = 100
COOLDOWN_TIME = 600

last_marked_time = {}

# ==============================
# FPS variables
# ==============================
prev_time = 0

# ==============================
# Open webcam
# ==============================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Camera not accessible")
    exit()

# ==============================
# CSV file
# ==============================
if not os.path.exists("attendance.csv"):
    with open("attendance.csv", "w") as f:
        f.write("Date,Name,Time,Duration,Status\n")

print("🚀 AI Classroom Attendance System Started")
print("Press 'q' to quit")

# ==============================
# MAIN LOOP
# ==============================
while True:

    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame (increase FPS)
    frame = cv2.resize(frame, (640,480))

    # ==============================
    # FPS calculation
    # ==============================
    curr_time = time.time()
    fps = int(1/(curr_time-prev_time)) if prev_time!=0 else 0
    prev_time = curr_time

    # ==============================
    # YOLO tracking
    # ==============================
    results = model.track(
        frame,
        classes=[0],
        conf=0.5,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False
    )

    annotated = frame.copy()
    current_time = time.time()

    for r in results:

        if r.boxes is None:
            continue

        for box in r.boxes:

            if box.id is None:
                continue

            yolo_id = int(box.id[0])

            # ==============================
            # Assign stable ID
            # ==============================
            if yolo_id not in id_map:

                id_map[yolo_id] = next_person_id

                track_meta[next_person_id] = {
                    "first_seen": current_time,
                    "last_seen": current_time,
                    "frame_count": 1
                }

                next_person_id += 1

            person_id = id_map[yolo_id]

            # ==============================
            # Update tracking info
            # ==============================
            if person_id not in track_meta:
                track_meta[person_id] = {
                    "first_seen": current_time,
                    "last_seen": current_time,
                    "frame_count": 1
                }
            else:
                track_meta[person_id]["last_seen"] = current_time
                track_meta[person_id]["frame_count"] += 1
            

            x1,y1,x2,y2 = map(int,box.xyxy[0])

            # ==============================
            # Smooth bounding box
            # ==============================
            alpha = 0.8

            if "prev_box" not in track_meta[person_id]:
                track_meta[person_id]["prev_box"] = (x1,y1,x2,y2)

            px1,py1,px2,py2 = track_meta[person_id]["prev_box"]

            x1 = int(alpha*px1 + (1-alpha)*x1)
            y1 = int(alpha*py1 + (1-alpha)*y1)
            x2 = int(alpha*px2 + (1-alpha)*x2)
            y2 = int(alpha*py2 + (1-alpha)*y2)

            track_meta[person_id]["prev_box"] = (x1,y1,x2,y2)

            # ==============================
            # FACE RECOGNITION (PERIODIC)
            # ==============================

            frames = track_meta[person_id]["frame_count"]

            # run recognition periodically
            if frames % 40 == 0:

                try:
                    face = frame[max(0,y1):min(frame.shape[0],y2),
                    max(0,x1):min(frame.shape[1],x2)]
                    predicted = recognize_face(face, face_db)
                except:
                    predicted = "Unknown"

                name_history[person_id].append(predicted)

                # keep last 5 predictions
                if len(name_history[person_id]) > 5:
                    name_history[person_id].pop(0)

                 # majority vote
                final_name = max(set(name_history[person_id]), key=name_history[person_id].count)

                person_names[person_id] = final_name

            name = person_names.get(person_id, "Detecting...")

            # ==============================
            # Attendance timing
            # ==============================
            duration = current_time-track_meta[person_id]["first_seen"]
            frames = track_meta[person_id]["frame_count"]

            if duration < MIN_TIME or frames < MIN_FRAMES:

                status_text[person_id] = f"Detecting {int(duration)}s"
                color = (0,255,255)

            else:

                status_text[person_id] = "Attendance Marked"
                color = (0,255,0)

            label = f"{name} | {status_text[person_id]}"

            # ==============================
            # Draw box
            # ==============================
            cv2.rectangle(annotated,(x1,y1),(x2,y2),color,2)

            cv2.putText(
                annotated,
                label,
                (x1,y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

            # ==============================
            # Attendance marking
            # ==============================
            can_mark=False

            if person_id not in last_marked_time:
                can_mark=True

            elif current_time-last_marked_time[person_id] >= COOLDOWN_TIME:
                can_mark=True

            if duration>=MIN_TIME and frames>=MIN_FRAMES and can_mark and name != "Unknown":

                last_marked_time[person_id]=current_time

                now=datetime.now()

                with open("attendance.csv","a") as f:

                    f.write(
                        f"{now.strftime('%Y-%m-%d')},"
                        f"{name},"
                        f"{now.strftime('%H:%M:%S')},"
                        f"{int(duration)}s,"
                        f"Present\n"
                    )

                print(f"✅ Attendance marked for {name}")

    # ==============================
    # Remove inactive tracks
    # ==============================
    remove_ids=[]

    for pid,meta in track_meta.items():

        if current_time-meta["last_seen"]>TRACK_TIMEOUT:
            remove_ids.append(pid)

    for pid in remove_ids:

        track_meta.pop(pid,None)
        person_names.pop(pid,None)

    # ==============================
    # Show FPS
    # ==============================
    cv2.putText(
        annotated,
        f"FPS: {fps}",
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    cv2.imshow("AI Attendance System",annotated)

    if cv2.waitKey(1)&0xFF==ord('q'):
        break

# ==============================
# CLEANUP
# ==============================
cap.release()
cv2.destroyAllWindows()

print("🛑 System stopped")