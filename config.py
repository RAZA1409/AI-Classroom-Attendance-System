"""
AI Classroom Attendance System - Centralized Configuration
===========================================================
Defines all operational thresholds, model specifications, file paths,
and runtime settings with dynamic environment-variable overrides.
"""

import os


class SystemConfig:
    """
    Central configuration container. Evaluates environment variables
    upon initialization to support runtime and test overrides.
    """

    def __init__(self):
        # --- Camera & Display ---
        self.CAMERA_INDEX: int = int(os.getenv("CAMERA_INDEX", "0"))
        self.FRAME_WIDTH: int = int(os.getenv("FRAME_WIDTH", "640"))
        self.FRAME_HEIGHT: int = int(os.getenv("FRAME_HEIGHT", "480"))

        # --- YOLO Person Detection & Tracking ---
        self.YOLO_MODEL_PATH: str = os.getenv("YOLO_MODEL_PATH", "yolov8n.pt")
        self.YOLO_CONF: float = float(os.getenv("YOLO_CONF", "0.5"))
        self.YOLO_TRACKER: str = os.getenv("YOLO_TRACKER", "bytetrack.yaml")
        self.TRACK_TIMEOUT: float = float(os.getenv("TRACK_TIMEOUT", "2.0"))
        self.EMA_ALPHA: float = float(os.getenv("EMA_ALPHA", "0.8"))

        # --- Face Detection & Recognition Pipeline ---
        self.FACE_DETECTOR_BACKEND: str = os.getenv("FACE_DETECTOR_BACKEND", "opencv")
        self.FACE_RECOGNITION_MODEL: str = os.getenv("FACE_RECOGNITION_MODEL", "Facenet512")
        self.RECOGNITION_THRESHOLD: float = float(os.getenv("RECOGNITION_THRESHOLD", "0.6"))
        self.RECOGNITION_INTERVAL: int = int(os.getenv("RECOGNITION_INTERVAL", "40"))
        self.VOTING_WINDOW_SIZE: int = int(os.getenv("VOTING_WINDOW_SIZE", "5"))

        # --- Face Quality Validation ---
        self.MIN_FACE_WIDTH: int = int(os.getenv("MIN_FACE_WIDTH", "40"))
        self.MIN_FACE_HEIGHT: int = int(os.getenv("MIN_FACE_HEIGHT", "40"))
        self.BLUR_THRESHOLD: float = float(os.getenv("BLUR_THRESHOLD", "30.0"))

        # --- Attendance Production Thresholds (Phase 2 Contracts) ---
        self.MIN_TIME_SECONDS: float = float(os.getenv("MIN_TIME_SECONDS", "5.0"))
        self.MIN_FRAMES: int = int(os.getenv("MIN_FRAMES", "100"))
        self.MAX_FRAME_GAP_SECONDS: float = float(os.getenv("MAX_FRAME_GAP_SECONDS", "0.25"))
        self.RECOVERY_WINDOW_SECONDS: float = float(os.getenv("RECOVERY_WINDOW_SECONDS", "3600.0"))

        # --- Storage & File Paths ---
        self.CSV_PATH: str = os.getenv("ATTENDANCE_CSV_PATH", "attendance.csv")
        self.STUDENTS_DIR: str = os.getenv("STUDENTS_DIR", "students")
        self.DATABASE_PATH: str = os.getenv("FACE_DATABASE_PATH", "face_embeddings.pkl")
        self.SESSION_MARKER_PATH: str = os.getenv("SESSION_MARKER_PATH", ".active_session.json")
        self.SESSION_ID: str = os.getenv("ATTENDANCE_SESSION_ID", "")

        # --- Logging ---
        self.LOG_DIR: str = os.getenv("LOG_DIR", "logs")
        self.LOG_FILE: str = os.getenv("LOG_FILE", "attendance.log")
        self.LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
        self.LOG_MAX_BYTES: int = int(os.getenv("LOG_MAX_BYTES", str(5 * 1024 * 1024)))  # 5 MB
        self.LOG_BACKUP_COUNT: int = int(os.getenv("LOG_BACKUP_COUNT", "3"))

    def reload(self):
        """Re-read configuration from environment variables."""
        self.__init__()


# Global default configuration instance
config = SystemConfig()
