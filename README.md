# AI Classroom Attendance System (Phase-1)

A real-time AI-powered classroom attendance system using **YOLOv8**, **OpenCV**, and **Python**.

This project automatically detects students, tracks them across frames, and marks attendance only after a **time-based confirmation**, reducing false positives.

---

## 🚀 Features (Phase-1)

- 🎯 Real-time person detection using YOLOv8
- 🔄 Multi-person tracking with stable IDs (session-based)
- ⏱️ Time-based attendance confirmation (minimum 5 seconds)
- 🟢 Visual status overlay:
  - Detecting
  - Attendance Marked
- 📊 FPS counter for performance monitoring
- 🧾 Automatic CSV logging with:
  - Date
  - Person ID
  - Time
  - Confirmation Duration
  - Attendance Status
- ❌ Duplicate attendance prevention in a single session

---

## 🧠 How It Works

1. Webcam captures live video
2. YOLOv8 detects and tracks persons
3. Each person is assigned a **stable session ID**
4. Attendance is marked only if the person stays visible for ≥ 5 seconds
5. Attendance is saved to `attendance.csv`

---

## 📂 CSV Format

```text
Date,Person_ID,Time,Duration,Status
2026-02-06,1,23:08:31,5s,Present