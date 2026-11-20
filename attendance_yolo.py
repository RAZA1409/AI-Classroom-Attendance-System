"""
AI Classroom Attendance System - Main Live Camera & Tracking Pipeline
=====================================================================
Integrates YOLOv8 body tracking with FaceNet512 facial recognition,
EMA display smoothing, and AttendanceManager session management.
"""

import os
import time
from collections import defaultdict
from typing import Tuple, Optional, List, Dict
import cv2
from ultralytics import YOLO

from config import config
from logger import get_logger
from face_database import load_database, recognize_face
from attendance_manager import AttendanceManager

logger = get_logger("attendance_yolo")


def verify_model_weights(model_path: str = config.YOLO_MODEL_PATH) -> bool:
    """
    Verify that YOLO model weights file exists locally before instantiation,
    preventing unintended automatic network downloads.
    """
    if not model_path or not os.path.exists(model_path):
        logger.error(
            "YOLO model weights '%s' not found. Please provide the model weights file "
            "in the project root to prevent unexpected network downloads.",
            model_path,
        )
        return False
    return True


def process_box_coordinates(
    raw_xyxy: Tuple[float, float, float, float],
    frame_shape: Tuple[int, int],
    prev_smoothed: Optional[Tuple[int, int, int, int]] = None,
    alpha: float = config.EMA_ALPHA,
) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[Tuple[int, int, int, int]]]:
    """
    Process detector bounding box into clamped raw coordinates (for recognition)
    and EMA-smoothed coordinates (for display).

    :param raw_xyxy: (x1, y1, x2, y2) from object detector.
    :param frame_shape: (height, width) of the video frame.
    :param prev_smoothed: Previous frame's smoothed display coordinates, if any.
    :param alpha: EMA weight factor for previous box (default: 0.8).
    :return: (raw_clamped_box, smoothed_display_box) or (None, None) if box is invalid.
    """
    frame_h, frame_w = frame_shape[:2]
    x1, y1, x2, y2 = map(int, raw_xyxy)

    rx1 = max(0, min(frame_w - 1, x1))
    ry1 = max(0, min(frame_h - 1, y1))
    rx2 = max(0, min(frame_w, x2))
    ry2 = max(0, min(frame_h, y2))

    if rx2 <= rx1 or ry2 <= ry1:
        return None, None

    raw_box = (rx1, ry1, rx2, ry2)

    if prev_smoothed is None:
        smoothed_box = raw_box
    else:
        px1, py1, px2, py2 = prev_smoothed
        sx1 = int(alpha * px1 + (1.0 - alpha) * rx1)
        sy1 = int(alpha * py1 + (1.0 - alpha) * ry1)
        sx2 = int(alpha * px2 + (1.0 - alpha) * rx2)
        sy2 = int(alpha * py2 + (1.0 - alpha) * ry2)
        smoothed_box = (sx1, sy1, sx2, sy2)

    return raw_box, smoothed_box


def cleanup_stale_tracks(
    track_meta: Dict[int, dict],
    id_map: Dict[int, int],
    person_names: Dict[int, str],
    name_history: Dict[int, list],
    smoothed_boxes: Dict[int, tuple],
    current_time: float,
    timeout: float = config.TRACK_TIMEOUT,
) -> List[int]:
    """
    Purge tracks inactive for more than timeout seconds.
    Removes person_id from all state dicts and eliminates reverse mappings
    in id_map to prevent memory leaks and recycled ID collisions.

    Returns:
        List of expired person_ids that were removed.
    """
    remove_ids = []
    for pid, meta in track_meta.items():
        if current_time - meta.get("last_seen", current_time) > timeout:
            remove_ids.append(pid)

    for pid in remove_ids:
        # Purge reverse mapping from id_map
        yolo_ids_to_remove = [yid for yid, mapped_pid in id_map.items() if mapped_pid == pid]
        for yid in yolo_ids_to_remove:
            id_map.pop(yid, None)

        track_meta.pop(pid, None)
        person_names.pop(pid, None)
        name_history.pop(pid, None)
        smoothed_boxes.pop(pid, None)

    return remove_ids


def assign_or_update_track(
    yolo_id: int,
    id_map: Dict[int, int],
    track_meta: Dict[int, dict],
    current_time: float,
    next_person_id: int,
) -> Tuple[int, int, bool]:
    """
    Assign a stable person_id to a YOLO tracker ID or update an active track.

    :param yolo_id: Tracker ID emitted by YOLO/ByteTrack.
    :param id_map: Dict mapping yolo_tracker_id -> person_id.
    :param track_meta: Dict mapping person_id -> metadata (first_seen, last_seen, frame_count).
    :param current_time: Current frame timestamp.
    :param next_person_id: Next sequential person_id counter.
    :return: (person_id, updated_next_person_id, is_new_track)
    """
    if yolo_id not in id_map:
        person_id = next_person_id
        id_map[yolo_id] = person_id
        track_meta[person_id] = {
            "first_seen": current_time,
            "last_seen": current_time,
            "frame_count": 1,
        }
        return person_id, next_person_id + 1, True
    else:
        person_id = id_map[yolo_id]
        track_meta[person_id]["last_seen"] = current_time
        track_meta[person_id]["frame_count"] += 1
        return person_id, next_person_id, False


def main():
    # ==============================
    # 1. Preflight Validation (Task B6)
    # ==============================
    if not verify_model_weights(config.YOLO_MODEL_PATH):
        print(f"❌ Error: YOLO model weights '{config.YOLO_MODEL_PATH}' not found locally.")
        exit(1)

    logger.info("Loading YOLO model: %s", config.YOLO_MODEL_PATH)
    model = YOLO(config.YOLO_MODEL_PATH)

    logger.info("Loading face database: %s", config.DATABASE_PATH)
    face_db = load_database(config.DATABASE_PATH)
    if not face_db:
        logger.warning("Face database is empty or not found. All faces will be classified as 'Unknown'.")

    # ==============================
    # 2. Tracking State & Structures (Task B7)
    # ==============================
    id_map = {}            # yolo_tracker_id -> person_id
    next_person_id = 1

    track_meta = {}        # person_id -> {first_seen, last_seen, frame_count}
    person_names = {}      # person_id -> current_majority_name
    name_history = defaultdict(list)  # person_id -> list of last N predictions
    smoothed_boxes = {}    # person_id -> (sx1, sy1, sx2, sy2)

    # ==============================
    # 3. Attendance Manager Initialization
    # ==============================
    attendance_mgr = AttendanceManager(
        csv_path=config.CSV_PATH,
        session_id=config.SESSION_ID if config.SESSION_ID else None,
        min_time_seconds=config.MIN_TIME_SECONDS,
        min_frames=config.MIN_FRAMES,
        max_frame_gap_seconds=config.MAX_FRAME_GAP_SECONDS,
        recovery_window_seconds=config.RECOVERY_WINDOW_SECONDS,
        session_marker_path=config.SESSION_MARKER_PATH,
    )

    # ==============================
    # 4. Open Webcam
    # ==============================
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        logger.error("Camera index %d not accessible. Please check camera connection.", config.CAMERA_INDEX)
        print(f"❌ Camera (index {config.CAMERA_INDEX}) not accessible.")
        exit(1)

    logger.info("AI Classroom Attendance System Started (Session ID: %s)", attendance_mgr.session_id)
    print("🚀 AI Classroom Attendance System Started")
    print(f"Session ID: {attendance_mgr.session_id}")
    print("Press 'q' to quit")

    prev_time = 0

    # ==============================
    # 5. Main Camera Loop
    # ==============================
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.warning("Camera stream terminated or frame could not be grabbed.")
            break

        # Resize frame for optimal FPS
        frame = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))
        frame_h, frame_w = frame.shape[:2]

        curr_time = time.time()
        fps = int(1.0 / (curr_time - prev_time)) if prev_time != 0 else 0
        prev_time = curr_time

        # Track people using YOLOv8
        results = model.track(
            frame,
            classes=[0],
            conf=config.YOLO_CONF,
            persist=True,
            tracker=config.YOLO_TRACKER,
            verbose=False,
        )

        annotated = frame.copy()
        current_time = time.time()

        # ==============================
        # Stale Track Cleanup & Tracker-ID Memory Purge (Task B7)
        # MUST run BEFORE processing current frame detections so recycled IDs start fresh!
        # ==============================
        cleanup_stale_tracks(
            track_meta=track_meta,
            id_map=id_map,
            person_names=person_names,
            name_history=name_history,
            smoothed_boxes=smoothed_boxes,
            current_time=current_time,
            timeout=config.TRACK_TIMEOUT,
        )

        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:
                if box.id is None:
                    continue

                yolo_id = int(box.id[0])

                # Assign or update stable person_id using production helper
                person_id, next_person_id, is_new_track = assign_or_update_track(
                    yolo_id=yolo_id,
                    id_map=id_map,
                    track_meta=track_meta,
                    current_time=current_time,
                    next_person_id=next_person_id,
                )

                # ==============================
                # Raw vs Smoothed Coordinates (Tasks B3)
                # ==============================
                raw_box, smoothed_box = process_box_coordinates(
                    raw_xyxy=box.xyxy[0],
                    frame_shape=(frame_h, frame_w),
                    prev_smoothed=smoothed_boxes.get(person_id),
                    alpha=config.EMA_ALPHA,
                )
                if raw_box is None:
                    continue

                smoothed_boxes[person_id] = smoothed_box
                rx1, ry1, rx2, ry2 = raw_box
                sx1, sy1, sx2, sy2 = smoothed_box

                # Recognition on raw person crop (Tasks B1 & B3)
                person_crop = frame[ry1:ry2, rx1:rx2]
                if person_crop.size == 0:
                    continue

                frames = track_meta[person_id]["frame_count"]

                # Run face recognition periodically using raw detector crop
                if frames % config.RECOGNITION_INTERVAL == 0:
                    predicted = recognize_face(
                        face_or_crop=person_crop,
                        database=face_db,
                        detector_backend=config.FACE_DETECTOR_BACKEND,
                        threshold=config.RECOGNITION_THRESHOLD,
                    )
                    name_history[person_id].append(predicted)

                    # Maintain sliding voting window
                    if len(name_history[person_id]) > config.VOTING_WINDOW_SIZE:
                        name_history[person_id].pop(0)

                    # Majority vote across recent window
                    final_name = max(set(name_history[person_id]), key=name_history[person_id].count)
                    person_names[person_id] = final_name

                name = person_names.get(person_id, "Detecting...")

                # ==============================
                # Attendance & Presence Evaluation
                # ==============================
                if name not in ("Unknown", "Detecting..."):
                    is_marked, status = attendance_mgr.update_presence(name, current_time)
                    color = (0, 255, 0) if is_marked else (0, 255, 255)
                elif name == "Unknown":
                    status = "Unknown"
                    color = (0, 0, 255)
                else:
                    status = "Detecting..."
                    color = (0, 255, 255)

                label = f"{name} | {status}"

                # Draw bounding box using smoothed display coordinates
                cv2.rectangle(annotated, (sx1, sy1), (sx2, sy2), color, 2)
                cv2.putText(
                    annotated,
                    label,
                    (sx1, max(20, sy1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

        # Show FPS
        cv2.putText(
            annotated,
            f"FPS: {fps}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        cv2.imshow("AI Attendance System", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    logger.info("AI Attendance System stopped safely.")
    print("🛑 System stopped")


if __name__ == "__main__":
    main()
