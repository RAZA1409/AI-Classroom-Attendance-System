AI Classroom Attendance System (YOLOv8 + DeepFace)

An AI-powered real-time classroom attendance system that automatically detects students and records attendance using computer vision and face recognition.

The system uses YOLOv8 for person detection & tracking and DeepFace (Facenet512 embeddings) for face recognition, allowing it to identify students and mark attendance automatically.

🚀 Features
🎯 Real-time Detection

Detects people in the classroom using YOLOv8 with real-time webcam input.

🧠 Face Recognition

Identifies students using DeepFace with Facenet512 embeddings.

⚡ Fast Startup (Embedding Cache)

Student face embeddings are precomputed and stored, allowing the system to start instantly without recalculating embeddings every run.

🔄 Stable Multi-Person Tracking

Uses ByteTrack tracking to maintain consistent identities even when multiple people are present.

🗳️ Recognition Smoothing

Multiple recognition results are combined using majority voting, reducing identity flickering.

⏱️ Time-Based Attendance

Attendance is only marked if a student remains visible for a minimum duration.

🟢 Visual Status Overlay

Each detected student shows:

Detecting

Attendance Marked

📊 FPS Monitoring

Displays real-time FPS for performance monitoring.

🧾 Automatic CSV Attendance Log

Attendance records are automatically saved with:

Date

Student Name

Time

Duration

Status

❌ Duplicate Prevention

The system prevents duplicate attendance entries during the same session.

🧠 How It Works

1️⃣ The webcam captures live video frames.

2️⃣ YOLOv8 detects and tracks people in the frame.

3️⃣ Each tracked person receives a stable session ID.

4️⃣ The system periodically extracts the face region and runs DeepFace recognition.

5️⃣ Recognition results are smoothed using majority voting.

6️⃣ If the student remains visible long enough, attendance is recorded.

7️⃣ Attendance is saved to attendance.csv.

📂 Project Structure
AI_Classroom_Attendance
│
├── attendance_yolo.py        # Main attendance system
├── face_database.py          # Face recognition functions
├── build_face_database.py    # Generates face embeddings
├── yolov8n.pt                # YOLO model
│
├── students/                 # Student face dataset
│   ├── raza/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │
│   ├── shivansh/
│       ├── img1.jpg
│
├── attendance.csv            # Generated attendance log
├── face_embeddings.pkl       # Cached face embeddings (generated)
└── README.md
📊 Attendance CSV Format
Date,Name,Time,Duration,Status
2026-03-14,raza,10:32:41,6s,Present
⚙️ Installation
1️⃣ Clone the repository
git clone https://github.com/RAZA1409/AI_Classroom_Attendance.git
cd AI_Classroom_Attendance
2️⃣ Create virtual environment
python -m venv ai_env

Activate it:

Windows

ai_env\Scripts\activate
3️⃣ Install dependencies
pip install ultralytics opencv-python deepface numpy pandas
🧑‍🎓 Add Students

Add student images inside the students folder.

Example:

students/
 ├── raza/
 │   ├── img1.jpg
 │   ├── img2.jpg
 │
 ├── shivansh/
     ├── img1.jpg

Use 5-10 clear images per student for better accuracy.

🧠 Build Face Database

Before running the system, generate face embeddings:

python build_face_database.py

This creates:

face_embeddings.pkl

which allows fast face recognition during runtime.

▶️ Run the Attendance System
python attendance_yolo.py

Press Q to stop the system.

📌 Technologies Used

Python

YOLOv8 (Ultralytics)

DeepFace

Facenet512

OpenCV

NumPy

🔮 Future Improvements

Planned upgrades for the project:

Face embedding cache optimization

Identity locking for more stable recognition

Higher FPS performance optimization

Web dashboard for attendance analytics

Face dataset auto-capture tool

📜 License

This project is for educational and research purposes.