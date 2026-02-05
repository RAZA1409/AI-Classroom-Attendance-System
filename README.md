# AI Classroom Attendance System 🎓🤖

An AI-powered real-time classroom attendance system using YOLOv8 and OpenCV.

## 🚀 Features
- Real-time person detection using YOLOv8
- Unique person tracking with IDs
- Time-based attendance confirmation (avoids false marking)
- Visual status overlay (Detecting → Attendance Marked)
- CSV-based attendance logging

## 🛠 Tech Stack
- Python
- YOLOv8 (Ultralytics)
- OpenCV
- NumPy

## 📸 Demo
> Real-time webcam-based detection with attendance confirmation.

## 📂 Project Structure
- `attendance_yolo.py` – Main attendance logic
- `yolo_test.py` – YOLO testing script
- `project_notes.md` – Future upgrade ideas
- `attendance.csv` – Attendance output (ignored in Git)

## ▶ How to Run
```bash
pip install ultralytics opencv-python
python attendance_yolo.py