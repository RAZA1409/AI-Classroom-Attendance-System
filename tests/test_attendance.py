"""
Automated Unit Tests for Phase 2: Attendance Correctness, Deduplication & Data Integrity
========================================================================================
All tests run in isolated temporary directories using synthetic data.
No webcam, display, GPU, or real student biometric data is required.
"""

import os
import sys
import csv
import json
import time
import shutil
import tempfile
import unittest

# Ensure project root is in Python search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from attendance_manager import (
    AttendanceManager,
    normalize_student_name,
    migrate_csv_schema_if_needed,
    resolve_session_id,
    write_active_session_marker,
)
from capture_student_faces import (
    validate_and_sanitize_student_name,
    check_existing_student_collision,
    get_next_image_index,
)


class TestAttendanceSystem(unittest.TestCase):

    def setUp(self):
        """Create a dedicated temporary directory for each test."""
        self.test_dir = tempfile.mkdtemp(prefix="test_attendance_")
        self.csv_path = os.path.join(self.test_dir, "test_attendance.csv")
        self.marker_path = os.path.join(self.test_dir, ".test_session.json")

    def tearDown(self):
        """Clean up temporary directory and all test artifacts."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def _read_csv_rows(self, path=None):
        target = path or self.csv_path
        if not os.path.exists(target):
            return []
        with open(target, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            return list(reader)

    # =============================================================
    # 1. ELIGIBILITY TESTS (Production Defaults: 5.0s AND 100 frames)
    # =============================================================

    def test_eligibility_production_thresholds_pass(self):
        """Verify that satisfying BOTH production thresholds (5.0s AND 100 frames) records attendance."""
        mgr = AttendanceManager(
            csv_path=self.csv_path,
            session_id="session_elig_prod",
            session_marker_path=self.marker_path,
            # Uses production defaults: min_time_seconds=5.0, min_frames=100, max_frame_gap_seconds=0.25
        )

        sim_time = 1000.0
        # 105 frames at 50ms intervals = 5.2 seconds of real time
        for _ in range(105):
            mgr.update_presence("Grace", current_time=sim_time)
            sim_time += 0.05

        rows = self._read_csv_rows()
        self.assertEqual(len(rows), 1, "Expected exactly 1 record for student satisfying production defaults")
        self.assertEqual(rows[0][1], "Grace")
        self.assertEqual(rows[0][4], "Present")
        self.assertEqual(rows[0][5], "session_elig_prod")

    def test_eligibility_insufficient_frames_rejected(self):
        """Verify that 12 frames / approximately 1.1s is rejected under production thresholds."""
        mgr = AttendanceManager(
            csv_path=self.csv_path,
            session_id="session_elig_short",
            session_marker_path=self.marker_path,
            # Uses production defaults: min_time_seconds=5.0, min_frames=100
        )

        sim_time = 1000.0
        for _ in range(12):
            mgr.update_presence("Frank", current_time=sim_time)
            sim_time += 0.1

        rows = self._read_csv_rows()
        self.assertEqual(len(rows), 0, "12 frames / 1.1s must NOT trigger attendance under production thresholds")

    def test_eligibility_only_frames_satisfied_rejected(self):
        """Verify that satisfying 100 frames but insufficient duration (< 5.0s) is rejected."""
        mgr = AttendanceManager(
            csv_path=self.csv_path,
            session_id="session_elig_frames_only",
            session_marker_path=self.marker_path,
            min_time_seconds=5.0,
            min_frames=100,
            max_frame_gap_seconds=0.25,
        )

        sim_time = 1000.0
        # 105 frames at 10ms intervals = 1.05 seconds (< 5.0s)
        for _ in range(105):
            mgr.update_presence("QuickUser", current_time=sim_time)
            sim_time += 0.01

        rows = self._read_csv_rows()
        self.assertEqual(len(rows), 0, "105 frames in 1.05s must be rejected because duration < 5.0s")

    def test_eligibility_only_duration_satisfied_rejected(self):
        """Verify that satisfying duration (>= 5.0s) but insufficient frames (< 100 frames) is rejected."""
        mgr = AttendanceManager(
            csv_path=self.csv_path,
            session_id="session_elig_time_only",
            session_marker_path=self.marker_path,
            min_time_seconds=5.0,
            min_frames=100,
            max_frame_gap_seconds=0.25,
        )

        sim_time = 1000.0
        # 30 frames at 200ms intervals = 6.0 seconds (< 100 frames)
        for _ in range(30):
            mgr.update_presence("SlowUser", current_time=sim_time)
            sim_time += 0.20

        rows = self._read_csv_rows()
        self.assertEqual(len(rows), 0, "6.0s across only 30 frames must be rejected because frames < 100")

    # =============================================================
    # 2. CSV SCHEMA MIGRATION & RECTANGULAR INTEGRITY TESTS
    # =============================================================

    def test_csv_legacy_migration_and_rectangular_integrity(self):
        """Migrate a legacy 5-column CSV to 6 columns, preserve records, and verify pandas/DictReader."""
        # Create legacy 5-column file
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Date", "Name", "Time", "Duration", "Status"])
            writer.writerow(["2026-08-25", "legacy_student_1", "16:36:52", "17s", "Present"])
            writer.writerow(["2026-08-25", "legacy_student_2", "16:37:44", "8s", "Present"])

        # Initializing manager must trigger safe migration
        mgr = AttendanceManager(
            csv_path=self.csv_path,
            session_id="session_mig_test",
            session_marker_path=self.marker_path,
            min_time_seconds=0.1,
            min_frames=2,
            max_frame_gap_seconds=0.25,
        )

        # Verify backup was created
        backup_path = self.csv_path + ".bak"
        self.assertTrue(os.path.exists(backup_path), "Backup copy .bak must be created before migration")

        # Record a new student
        mgr.update_presence("new_student", current_time=1000.0)
        mgr.update_presence("new_student", current_time=1000.1)

        # 1. Verify with raw csv.reader
        with open(self.csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            self.assertEqual(header, ["Date", "Name", "Time", "Duration", "Status", "Session_ID"])
            all_rows = list(reader)

        self.assertEqual(len(all_rows), 3)
        for idx, row in enumerate(all_rows):
            self.assertEqual(len(row), 6, f"Row {idx} must have exactly 6 columns")

        # Legacy rows backfilled with empty string
        self.assertEqual(all_rows[0][5], "")
        self.assertEqual(all_rows[1][5], "")
        # New row has Session_ID
        self.assertEqual(all_rows[2][5], "session_mig_test")

        # 2. Verify with csv.DictReader
        with open(self.csv_path, "r", newline="", encoding="utf-8") as f:
            dict_reader = csv.DictReader(f)
            self.assertEqual(dict_reader.fieldnames, ["Date", "Name", "Time", "Duration", "Status", "Session_ID"])
            dict_rows = list(dict_reader)
            self.assertEqual(len(dict_rows), 3)
            # Ensure no None key exists (which happens on ragged rows)
            for r in dict_rows:
                self.assertNotIn(None, r, "csv.DictReader must not have None key on rectangular rows")
                self.assertIn("Session_ID", r)

        # 3. Verify with pandas if available
        try:
            import pandas as pd
            df = pd.read_csv(self.csv_path)
            self.assertEqual(len(df.columns), 6)
            self.assertEqual(list(df.columns), ["Date", "Name", "Time", "Duration", "Status", "Session_ID"])
            self.assertEqual(len(df), 3)
        except ImportError:
            pass

    def test_csv_empty_and_corrupt_files_handled_safely(self):
        """Empty or malformed files are handled safely without crashing."""
        # Test empty file (0 bytes)
        with open(self.csv_path, "w", encoding="utf-8") as f:
            pass
        success, msg = migrate_csv_schema_if_needed(self.csv_path)
        self.assertTrue(success)

        # Test malformed file with uneven columns
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            f.write("Col1,Col2\nrow1\n")
        success, msg = migrate_csv_schema_if_needed(self.csv_path)
        self.assertTrue(success)
        # Verify file now has standard 6-column header
        with open(self.csv_path, "r", newline="", encoding="utf-8") as f:
            header = next(csv.reader(f))
            self.assertEqual(header, ["Date", "Name", "Time", "Duration", "Status", "Session_ID"])

    # =============================================================
    # 3. SESSION RECOVERY ACROSS APPLICATION RESTARTS
    # =============================================================

    def test_session_recovery_automatic_without_env_var(self):
        """Restart without explicit env var resumes the active session if within 60-minute window."""
        t0 = 1000000.0
        # Run 1: initial launch
        mgr1 = AttendanceManager(
            csv_path=self.csv_path,
            session_id=None,  # No explicit session ID
            session_marker_path=self.marker_path,
            recovery_window_seconds=3600.0,
            current_time=t0,
            min_time_seconds=0.1,
            min_frames=2,
            max_frame_gap_seconds=0.25,
        )
        sid_1 = mgr1.session_id
        self.assertFalse(mgr1.was_resumed)

        # Mark Alice
        mgr1.update_presence("Alice", current_time=t0)
        mgr1.update_presence("Alice", current_time=t0 + 0.1)

        rows1 = self._read_csv_rows()
        self.assertEqual(len(rows1), 1)

        # Run 2: Process restarts 10 minutes later (600s < 3600s) during ongoing class
        t1 = t0 + 600.0
        mgr2 = AttendanceManager(
            csv_path=self.csv_path,
            session_id=None,  # No explicit session ID
            session_marker_path=self.marker_path,
            recovery_window_seconds=3600.0,
            current_time=t1,
            min_time_seconds=0.1,
            min_frames=2,
            max_frame_gap_seconds=0.25,
        )

        # Session must be resumed!
        self.assertTrue(mgr2.was_resumed, "Session should be automatically resumed within recovery window")
        self.assertEqual(mgr2.session_id, sid_1, "Session ID must match the active session on restart")

        # Alice is seen again in the resumed session
        for dt_step in range(10):
            is_marked, status = mgr2.update_presence("Alice", current_time=t1 + (dt_step * 0.1))
            self.assertTrue(is_marked)
            self.assertEqual(status, "Attendance Marked")

        rows2 = self._read_csv_rows()
        self.assertEqual(len(rows2), 1, "Alice must NOT be duplicated in CSV after restart")

    def test_session_expired_creates_new_session(self):
        """Restart after recovery window (> 60 minutes) generates a new session ID."""
        t0 = 1000000.0
        write_active_session_marker(self.marker_path, "session_old", t0)

        # Restart 2 hours later (7200s > 3600s)
        t_restart = t0 + 7200.0
        sid, was_resumed = resolve_session_id(
            explicit_session_id=None,
            marker_path=self.marker_path,
            recovery_window_seconds=3600.0,
            current_time=t_restart,
        )
        self.assertFalse(was_resumed, "Expired session must not be resumed")
        self.assertNotEqual(sid, "session_old")

    def test_session_explicit_id_overrides_marker(self):
        """Explicit session ID always overrides any active session marker."""
        t0 = 1000000.0
        write_active_session_marker(self.marker_path, "session_marker_val", t0)

        sid, was_resumed = resolve_session_id(
            explicit_session_id="class_math_101",
            marker_path=self.marker_path,
            recovery_window_seconds=3600.0,
            current_time=t0 + 100.0,
        )
        self.assertEqual(sid, "class_math_101")
        self.assertFalse(was_resumed)

    def test_session_corrupt_marker_handled_safely(self):
        """Corrupted JSON in session marker is handled safely by generating a new session ID."""
        with open(self.marker_path, "w", encoding="utf-8") as f:
            f.write("{invalid json content!!!")

        sid, was_resumed = resolve_session_id(
            explicit_session_id=None,
            marker_path=self.marker_path,
            recovery_window_seconds=3600.0,
            current_time=1000000.0,
        )
        self.assertFalse(was_resumed)
        self.assertTrue(sid.startswith("session_"))

    # =============================================================
    # 4. PRESENCE TIMING & GAP POLICY TESTS
    # =============================================================

    def test_timing_normal_frame_intervals_accumulate(self):
        """Normal inter-frame intervals (dt <= 0.25s) accumulate measured time correctly."""
        mgr = AttendanceManager(
            csv_path=self.csv_path,
            session_id="session_timing",
            session_marker_path=self.marker_path,
            min_time_seconds=10.0,
            min_frames=100,
            max_frame_gap_seconds=0.25,
        )

        sim_time = 1000.0
        # 10 frames at 0.05s interval = 9 steps of 0.05s = 0.45s elapsed
        for _ in range(10):
            mgr.update_presence("Student_T", current_time=sim_time)
            sim_time += 0.05

        presence = mgr.student_presence[normalize_student_name("Student_T")]
        self.assertEqual(presence["qualifying_frames"], 10)
        self.assertAlmostEqual(presence["total_seconds"], 0.45, places=4)

    def test_timing_camera_freeze_does_not_inflate_duration(self):
        """Simulated 2.5-second camera freeze (dt > 0.25s) must NOT credit 2.5s of presence."""
        mgr = AttendanceManager(
            csv_path=self.csv_path,
            session_id="session_freeze",
            session_marker_path=self.marker_path,
            min_time_seconds=10.0,
            min_frames=100,
            max_frame_gap_seconds=0.25,
        )

        # Frame 1 at t=0
        mgr.update_presence("User_Freeze", current_time=0.0)
        # Camera freezes / student absent for 2.5 seconds. Frame 2 at t=2.5s
        mgr.update_presence("User_Freeze", current_time=2.5)

        presence = mgr.student_presence[normalize_student_name("User_Freeze")]
        self.assertEqual(presence["qualifying_frames"], 2)
        # Total seconds must NOT have the 2.5s added!
        self.assertEqual(presence["total_seconds"], 0.0, "2.5s gap must NOT be credited as presence duration")

        # Next normal frame at t=2.55s (dt = 0.05s <= 0.25s)
        mgr.update_presence("User_Freeze", current_time=2.55)
        self.assertEqual(presence["qualifying_frames"], 3)
        self.assertAlmostEqual(presence["total_seconds"], 0.05, places=4)

    def test_timing_negative_and_repeated_timestamps_ignored(self):
        """Negative time differences or duplicate timestamps do not corrupt or reduce duration."""
        mgr = AttendanceManager(
            csv_path=self.csv_path,
            session_id="session_clock_skew",
            session_marker_path=self.marker_path,
            min_time_seconds=10.0,
            min_frames=100,
            max_frame_gap_seconds=0.25,
        )

        mgr.update_presence("User_Clock", current_time=100.0)
        mgr.update_presence("User_Clock", current_time=100.1)  # +0.1s
        # Backward clock shift to 95.0s
        mgr.update_presence("User_Clock", current_time=95.0)

        presence = mgr.student_presence[normalize_student_name("User_Clock")]
        self.assertAlmostEqual(presence["total_seconds"], 0.1, places=4)

    # =============================================================
    # 5. IDENTITY NORMALIZATION & COLLISION SAFEGUARD TESTS
    # =============================================================

    def test_identity_case_insensitive_matching(self):
        """'Raza', 'raza', and ' RAZA ' map to the exact same identity and deduplicate cleanly."""
        mgr = AttendanceManager(
            csv_path=self.csv_path,
            session_id="session_case_test",
            session_marker_path=self.marker_path,
            min_time_seconds=0.1,
            min_frames=2,
            max_frame_gap_seconds=0.25,
        )

        sim_time = 1000.0
        # Mark under 'Raza'
        mgr.update_presence("Raza", current_time=sim_time)
        mgr.update_presence("Raza", current_time=sim_time + 0.1)

        rows = self._read_csv_rows()
        self.assertEqual(len(rows), 1)

        # Check under lowercase and uppercase
        self.assertTrue(mgr.is_marked("raza"))
        self.assertTrue(mgr.is_marked("RAZA"))
        self.assertTrue(mgr.is_marked("  raza  "))

        # Detection under 'raza' returns marked
        is_marked, status = mgr.update_presence("raza", current_time=sim_time + 0.2)
        self.assertTrue(is_marked)
        self.assertEqual(status, "Attendance Marked")

        rows2 = self._read_csv_rows()
        self.assertEqual(len(rows2), 1, "Case variation must NOT produce duplicate attendance")

    def test_identity_isolation_between_different_students(self):
        """Two different students accumulate presence completely independently."""
        mgr = AttendanceManager(
            csv_path=self.csv_path,
            session_id="session_iso_test",
            session_marker_path=self.marker_path,
            min_time_seconds=0.5,
            min_frames=5,
            max_frame_gap_seconds=0.25,
        )

        sim_time = 1000.0
        # Student A seen for 3 frames (not eligible)
        for _ in range(3):
            mgr.update_presence("Student_A", current_time=sim_time)
            sim_time += 0.1

        # Student B seen for 6 frames (5 intervals of 0.1s = 0.5s, eligible)
        for _ in range(6):
            mgr.update_presence("Student_B", current_time=sim_time)
            sim_time += 0.1

        rows = self._read_csv_rows()
        self.assertEqual(len(rows), 1, "Only Student B should be marked present")
        self.assertEqual(rows[0][1], "Student_B")

        # Verify Student A did NOT receive Student B's frames
        norm_a = normalize_student_name("Student_A")
        self.assertEqual(mgr.student_presence[norm_a]["qualifying_frames"], 3)

    def test_identity_tracker_id_swap_does_not_duplicate(self):
        """Tracker ID reassignment does not duplicate attendance."""
        mgr = AttendanceManager(
            csv_path=self.csv_path,
            session_id="session_tracker_swap",
            session_marker_path=self.marker_path,
            min_time_seconds=0.2,
            min_frames=4,
            max_frame_gap_seconds=0.25,
        )

        sim_time = 1000.0
        # 2 frames under Tracker ID 1
        mgr.update_presence("Student_Swap", current_time=sim_time)
        mgr.update_presence("Student_Swap", current_time=sim_time + 0.05)

        # 3 frames under Tracker ID 9
        mgr.update_presence("Student_Swap", current_time=sim_time + 0.10)
        mgr.update_presence("Student_Swap", current_time=sim_time + 0.15)
        mgr.update_presence("Student_Swap", current_time=sim_time + 0.20)

        rows = self._read_csv_rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][1], "Student_Swap")

    def test_enrollment_safeguards_and_collisions(self):
        """Enrollment rejects invalid names, prevents collisions, and protects files from overwrite."""
        # 1. Reject invalid names
        self.assertEqual(validate_and_sanitize_student_name(""), "")
        self.assertEqual(validate_and_sanitize_student_name("   "), "")
        self.assertEqual(validate_and_sanitize_student_name("///"), "")
        self.assertEqual(validate_and_sanitize_student_name("___"), "")
        self.assertEqual(validate_and_sanitize_student_name("!@#$%^&*()"), "")

        # 2. Valid names
        self.assertEqual(validate_and_sanitize_student_name("John Doe"), "John_Doe")
        self.assertEqual(validate_and_sanitize_student_name("Jane-Doe"), "Jane-Doe")

        # 3. Collision check
        mock_students = os.path.join(self.test_dir, "mock_students")
        os.makedirs(os.path.join(mock_students, "Raza"), exist_ok=True)
        collides, canonical = check_existing_student_collision(mock_students, "raza")
        self.assertTrue(collides)
        self.assertEqual(canonical, "Raza")

        # 4. Overwrite protection
        folder = os.path.join(mock_students, "Raza")
        with open(os.path.join(folder, "Raza_0.jpg"), "w") as f:
            f.write("image 0")
        with open(os.path.join(folder, "Raza_1.jpg"), "w") as f:
            f.write("image 1")

        next_idx = get_next_image_index(folder, "Raza")
        self.assertEqual(next_idx, 2, "Next image index must be 2 to avoid overwriting 0 and 1")


if __name__ == "__main__":
    unittest.main()
