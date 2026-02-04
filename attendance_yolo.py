from ultralytics import YOLO
import cv2
import pandas as pd
from datetime import datetime

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

attendance = {}
start_time = datetime.now().strftime("%Y-%m-%d")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, classes=[0], conf=0.5, persist=True)


    for r in results:
        for box in r.boxes:
            person_id = int(box.id[0]) if box.id is not None else None

            if person_id is not None and person_id not in attendance:
                attendance[person_id] = datetime.now().strftime("%H:%M:%S")

    annotated = results[0].plot()
    cv2.imshow("AI Attendance System", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

from datetime import date

today = date.today().strftime("%Y-%m-%d")

rows = []
for pid, time in attendance.items():
    rows.append([today, pid, time])

df = pd.DataFrame(rows, columns=["Date", "Person_ID", "Time"])
df.to_csv("attendance.csv", index=False)

print("✅ Attendance saved with DATE to attendance.csv")

