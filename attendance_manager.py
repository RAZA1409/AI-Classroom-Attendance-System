"""
Attendance Manager Module
=========================
Handles attendance session management, presence duration accumulation,
eligibility enforcement, robust deduplication, safe CSV storage, and
legacy schema migration.

Decouples attendance logic from camera I/O and object tracking IDs.
"""

import os
import re
import csv
import json
import time
import shutil
import tempfile
from datetime import datetime
from typing import Dict, Tuple, Set, Optional


def normalize_student_name(name: str) -> str:
    """
    Canonical name-normalization function.
    Strips surrounding whitespace, converts to lowercase,
    and collapses multiple whitespace/underscore sequences into a single underscore.
    Returns empty string if name is invalid.
    """
    if not name or not isinstance(name, str):
        return ""
    cleaned = name.strip().lower()
    cleaned = re.sub(r'[\s_]+', '_', cleaned)
    return cleaned.strip('_')


def migrate_csv_schema_if_needed(csv_path: str) -> Tuple[bool, str]:
    """
    Ensure the CSV file at csv_path adheres to the 6-column schema:
    Date,Name,Time,Duration,Status,Session_ID

    If the file exists and has a legacy 5-column header (or missing Session_ID):
    - Creates a backup copy at csv_path + '.bak' before making any changes.
    - Rewrites the file with the 6-column header.
    - Preserves all original rows and backfills the missing Session_ID with an empty string.
    - Uses a temporary file and atomic replacement (os.replace).
    - If any error occurs, the original file is preserved untouched.

    Returns:
        (success: bool, status_message: str)
    """
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return True, "No migration needed (file absent or empty)"

    try:
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                return True, "Empty file"

            normalized_header = [col.strip().lower() for col in header]
            if len(header) >= 6 and "session_id" in normalized_header:
                return True, "Header already 6-column compliant"

            rows = list(reader)

        # Create backup before modification
        backup_path = csv_path + ".bak"
        shutil.copy2(csv_path, backup_path)

        # Prepare 6-column rows
        new_header = ["Date", "Name", "Time", "Duration", "Status", "Session_ID"]
        migrated_rows = []
        for row in rows:
            if not row:
                continue
            if len(row) == 5:
                migrated_rows.append(row + [""])
            elif len(row) >= 6:
                migrated_rows.append(row[:6])
            else:
                # Pad shorter malformed rows with empty strings
                migrated_rows.append(row + [""] * (6 - len(row)))

        # Write to temporary file in the same directory for atomic replace
        dir_name = os.path.dirname(os.path.abspath(csv_path))
        temp_fd, temp_path = tempfile.mkstemp(dir=dir_name, prefix="att_mig_", suffix=".tmp")
        try:
            with os.fdopen(temp_fd, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(new_header)
                for r in migrated_rows:
                    writer.writerow(r)
                f.flush()

            os.replace(temp_path, csv_path)
            return True, f"Migrated {len(migrated_rows)} rows to 6-column schema (backup: {backup_path})"
        except Exception as e:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
            return False, f"Migration atomic replace failed: {e}"

    except Exception as e:
        return False, f"Migration failed: {e}"


def resolve_session_id(
    explicit_session_id: Optional[str] = None,
    marker_path: str = ".active_session.json",
    recovery_window_seconds: float = 3600.0,
    current_time: Optional[float] = None,
) -> Tuple[str, bool]:
    """
    Resolve active session ID according to priority rules:
    1. Explicit session ID (takes precedence).
    2. Active session marker (.active_session.json) if valid and not expired.
    3. Generate new session ID (session_YYYYMMDD_HHMMSS).

    Returns:
        (session_id: str, was_resumed: bool)
    """
    if current_time is None:
        current_time = time.time()

    # Rule 1: Explicitly provided session ID always takes priority
    if explicit_session_id and str(explicit_session_id).strip():
        sid = str(explicit_session_id).strip()
        write_active_session_marker(marker_path, sid, current_time)
        return sid, False

    # Rule 2: Check active session marker
    if os.path.exists(marker_path) and os.path.getsize(marker_path) > 0:
        try:
            with open(marker_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, dict):
                sid = data.get("session_id", "").strip()
                last_active = float(data.get("last_active_time", 0.0))
                # Check expiration window (default: 60 minutes)
                if sid and 0.0 <= (current_time - last_active) <= recovery_window_seconds:
                    write_active_session_marker(marker_path, sid, current_time)
                    return sid, True
        except Exception:
            # Corrupted marker, invalid JSON, or unreadable: fallback to new session
            pass

    # Rule 3: Generate new session ID
    new_sid = datetime.fromtimestamp(current_time).strftime("session_%Y%m%d_%H%M%S")
    write_active_session_marker(marker_path, new_sid, current_time)
    return new_sid, False


def write_active_session_marker(marker_path: str, session_id: str, current_time: float) -> bool:
    """
    Atomically write active session marker containing session_id and timestamp.
    Contains zero student names, zero biometric data, zero PII.
    """
    try:
        dir_name = os.path.dirname(os.path.abspath(marker_path))
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        data = {
            "session_id": session_id,
            "last_active_time": current_time,
            "updated_at": datetime.fromtimestamp(current_time).strftime("%Y-%m-%d %H:%M:%S")
        }
        temp_fd, temp_path = tempfile.mkstemp(dir=dir_name, prefix=".sess_", suffix=".tmp")
        with os.fdopen(temp_fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.flush()
        os.replace(temp_path, marker_path)
        return True
    except Exception as e:
        print(f"⚠️ Warning: Failed to write session marker '{marker_path}': {e}")
        return False


class AttendanceManager:
    """
    Manages student presence tracking, attendance eligibility,
    session-based deduplication, and persistent CSV storage.
    """

    def __init__(
        self,
        csv_path: str = "attendance.csv",
        session_id: Optional[str] = None,
        min_time_seconds: float = 5.0,
        min_frames: int = 100,
        max_frame_gap_seconds: float = 0.25,
        recovery_window_seconds: float = 3600.0,
        session_marker_path: str = ".active_session.json",
        class_name: str = "General",
        current_time: Optional[float] = None,
    ):
        """
        Initialize the AttendanceManager.

        :param csv_path: Path to the attendance CSV file.
        :param session_id: Explicit class session ID. If None, resolved via marker/auto-generated.
        :param min_time_seconds: Minimum presence time in seconds required for eligibility (default 5.0).
        :param min_frames: Minimum qualifying frames required for eligibility (default 100).
        :param max_frame_gap_seconds: Maximum allowable inter-frame gap (default 0.25s). Gaps larger
                                      than this do not count towards presence duration.
        :param recovery_window_seconds: Time window (default 3600s = 60 min) for session recovery.
        :param session_marker_path: Path to local active session marker file.
        :param class_name: Optional class or subject identifier.
        :param current_time: Optional timestamp for deterministic initialization.
        """
        self.csv_path = csv_path
        self.min_time = float(min_time_seconds)
        self.min_frames = int(min_frames)
        self.max_frame_gap = float(max_frame_gap_seconds)
        self.recovery_window = float(recovery_window_seconds)
        self.session_marker_path = session_marker_path
        self.class_name = class_name

        init_time = current_time if current_time is not None else time.time()

        # Resolve session ID: explicit > valid marker > auto-generated
        self.session_id, self.was_resumed = resolve_session_id(
            explicit_session_id=session_id,
            marker_path=self.session_marker_path,
            recovery_window_seconds=self.recovery_window,
            current_time=init_time,
        )

        # In-memory tracking: set of (normalized_student_name, session_id)
        self.marked_records: Set[Tuple[str, str]] = set()

        # Track presence per student (normalized_name -> presence dict)
        self.student_presence: Dict[str, dict] = {}

        # 1. Migrate legacy CSV if needed before loading
        success, msg = migrate_csv_schema_if_needed(self.csv_path)
        if not success:
            print(f"⚠️ Warning: {msg}")

        # 2. Load existing records for persistent deduplication
        self._load_existing_records()

    def _load_existing_records(self) -> None:
        """
        Load historical records from CSV to prevent duplicate attendance
        if the application restarts within the same session.
        """
        if not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) == 0:
            return

        try:
            with open(self.csv_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if not header:
                    return

                for row in reader:
                    if not row or len(row) < 2:
                        continue
                    row_name = row[1].strip()
                    norm_name = normalize_student_name(row_name)
                    if not norm_name:
                        continue

                    # If row has Session_ID (column index 5)
                    if len(row) >= 6 and row[5].strip():
                        row_session = row[5].strip()
                        self.marked_records.add((norm_name, row_session))
                    else:
                        # Legacy fallback: associate with legacy session using date
                        row_date = row[0].strip()
                        self.marked_records.add((norm_name, f"legacy_{row_date}"))
                        self.marked_records.add((norm_name, row_date))
                        # Also protect under empty session string
                        self.marked_records.add((norm_name, ""))
        except Exception as e:
            print(f"⚠️ Warning: Could not read existing records from '{self.csv_path}': {e}")

    def is_marked(self, student_name: str, session_id: Optional[str] = None) -> bool:
        """
        Check if student has already been marked present in the given or current session.
        Case-insensitive and whitespace-normalized.
        """
        norm_name = normalize_student_name(student_name)
        if not norm_name or norm_name in ("unknown", "detecting"):
            return False
        sid = session_id if session_id is not None else self.session_id
        return (norm_name, sid) in self.marked_records

    def heartbeat(self, current_time: Optional[float] = None) -> None:
        """
        Update the active session marker's last_active_time to keep the session alive.
        """
        now = current_time if current_time is not None else time.time()
        write_active_session_marker(self.session_marker_path, self.session_id, now)

    def update_presence(self, student_name: str, current_time: Optional[float] = None) -> Tuple[bool, str]:
        """
        Update presence accumulation for a confirmed student identity and evaluate eligibility.

        :param student_name: Recognized student name.
        :param current_time: Timestamp of detection. Defaults to time.time().
        :return: (is_marked_or_eligible: bool, status_message: str)
        """
        if current_time is None:
            current_time = time.time()

        norm_name = normalize_student_name(student_name)
        if not norm_name or norm_name in ("unknown", "detecting"):
            return False, "Unknown"

        # Update heartbeat
        self.heartbeat(current_time)

        # If student is already marked for this session, return immediately
        if self.is_marked(student_name):
            return True, "Attendance Marked"

        # Initialize student presence on first observation
        if norm_name not in self.student_presence:
            self.student_presence[norm_name] = {
                "display_name": student_name.strip(),
                "first_seen": current_time,
                "last_seen": current_time,
                "last_frame_time": current_time,
                "total_seconds": 0.0,
                "qualifying_frames": 1,
            }
        else:
            rec = self.student_presence[norm_name]
            dt = current_time - rec["last_frame_time"]

            # Guard against negative or repeated timestamps
            if dt > 0:
                if dt <= self.max_frame_gap:
                    # Normal frame interval: add measured dt
                    rec["total_seconds"] += dt
                    rec["qualifying_frames"] += 1
                else:
                    # Gap detected (> max_frame_gap, e.g. absence or camera freeze):
                    # Do NOT add dt to total_seconds!
                    # Resume accumulation from this new valid detection
                    rec["qualifying_frames"] += 1

                rec["last_seen"] = current_time
                rec["last_frame_time"] = current_time

        presence = self.student_presence[norm_name]
        duration = presence["total_seconds"]
        frames = presence["qualifying_frames"]

        # Evaluate attendance eligibility against configured thresholds
        if round(duration, 4) >= round(self.min_time, 4) and frames >= self.min_frames:
            success = self.record_attendance(presence.get("display_name", student_name), duration, current_time)
            if success:
                return True, "Attendance Marked"

        return False, f"Detecting {int(duration)}s"

    def record_attendance(
        self,
        student_name: str,
        duration: float,
        timestamp: Optional[float] = None,
        status: str = "Present",
    ) -> bool:
        """
        Safely append an attendance record to CSV. Enforces deduplication.

        :param student_name: Name of the student.
        :param duration: Validated presence duration in seconds.
        :param timestamp: Unix timestamp. Defaults to current time.
        :param status: Attendance status, defaults to "Present".
        :return: True if recorded successfully, False if already marked or failed.
        """
        if self.is_marked(student_name):
            return False

        norm_name = normalize_student_name(student_name)
        if not norm_name or norm_name in ("unknown", "detecting"):
            return False

        now = datetime.fromtimestamp(timestamp) if timestamp else datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")
        duration_str = f"{int(duration)}s"

        dir_name = os.path.dirname(self.csv_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        file_exists = os.path.exists(self.csv_path) and os.path.getsize(self.csv_path) > 0

        try:
            with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["Date", "Name", "Time", "Duration", "Status", "Session_ID"])

                writer.writerow([date_str, student_name, time_str, duration_str, status, self.session_id])
                f.flush()

            # Record in normalized memory set
            self.marked_records.add((norm_name, self.session_id))
            print(f"✅ Attendance recorded: {student_name} (Session: {self.session_id})")
            return True
        except Exception as e:
            print(f"❌ Error writing attendance to '{self.csv_path}': {e}")
            return False

    def get_session_records(self) -> Dict[str, dict]:
        """
        Get all students marked in the current session.
        """
        result = {}
        for (norm_name, sid) in self.marked_records:
            if sid == self.session_id:
                presence = self.student_presence.get(norm_name, {})
                display_name = presence.get("display_name", norm_name)
                result[display_name] = {
                    "session_id": sid,
                    "duration": presence.get("total_seconds", 0.0),
                    "frames": presence.get("qualifying_frames", 0),
                    "marked": True,
                }
        return result
