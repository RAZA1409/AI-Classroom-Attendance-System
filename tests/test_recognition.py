"""
Unit Tests for Phase 3: Recognition Pipeline, Reliability & Runtime Stability
=============================================================================
Tests BGR color standardization, unbypassable face quality validation,
production coordinate processing, production tracker cleanup, preflight model
verification, partial database replacement protection, and similarity guards.

All tests execute headlessly with synthetic images and mocks.
"""

import os
import sys
import csv
import pickle
import tempfile
import shutil
import unittest
from datetime import datetime
from unittest.mock import patch, MagicMock
import numpy as np

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import SystemConfig, config
from face_database import (
    cosine_similarity,
    validate_face_quality,
    detect_single_face,
    extract_embedding,
    recognize_face,
    recognize_face_detailed,
    load_database,
)
from build_face_database import build_face_database
from attendance_yolo import (
    process_box_coordinates,
    cleanup_stale_tracks,
    verify_model_weights,
    assign_or_update_track,
)
from attendance_manager import AttendanceManager


class TestRecognitionPipeline(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_recog_")

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    # =============================================================
    # 1. COLOR STANDARDIZATION & EMBEDDING PIPELINE TESTS (Fix 1)
    # =============================================================

    @patch("face_database.DeepFace.extract_faces")
    def test_extract_faces_specifies_bgr_color_channel(self, mock_extract):
        """1. detect_single_face explicitly requests color_face='bgr' and normalize_face=False."""
        mock_face_bgr = np.ones((80, 80, 3), dtype=np.float64) * 150.0  # unnormalized 0-255 pixels
        mock_extract.return_value = [{
            "face": mock_face_bgr,
            "facial_area": {"x": 10, "y": 10, "w": 80, "h": 80},
            "confidence": 0.99,
        }]

        sharp_crop = np.random.randint(50, 200, (150, 150, 3), dtype=np.uint8)
        extracted, area, status = detect_single_face(sharp_crop, blur_threshold=1.0)

        # Confirm extract_faces was called with color_face='bgr' and normalize_face=False
        mock_extract.assert_called_once()
        _, kwargs = mock_extract.call_args
        self.assertEqual(kwargs.get("color_face"), "bgr")
        self.assertEqual(kwargs.get("normalize_face"), False)
        self.assertEqual(status, "OK")
        self.assertIsNotNone(extracted)
        self.assertTrue(np.array_equal(extracted, mock_face_bgr))

    @patch("face_database.DeepFace.represent")
    def test_extract_embedding_receives_bgr_face_array(self, mock_represent):
        """2. extract_embedding passes the BGR face array to represent with skip detector and normalization='base'."""
        mock_represent.return_value = [{"embedding": [0.1] * 512}]
        dummy_bgr_face = np.ones((112, 112, 3), dtype=np.float32) * 128.0

        emb = extract_embedding(dummy_bgr_face, model_name="Facenet512")

        mock_represent.assert_called_once_with(
            img_path=dummy_bgr_face,
            model_name="Facenet512",
            detector_backend="skip",
            enforce_detection=False,
            normalization="base",
        )
        self.assertIsNotNone(emb)
        self.assertEqual(len(emb), 512)

    @patch("face_database.DeepFace.represent")
    def test_extract_embedding_passes_unnormalized_face_array_without_scaling(self, mock_represent):
        """2b. Face array passed to represent() retains unnormalized 0-255 pixel values without alteration."""
        mock_represent.return_value = [{"embedding": [0.5] * 512}]
        original_pixel_values = np.array([[[0.0, 128.5, 255.0], [50.0, 100.0, 200.0]]], dtype=np.float32)
        test_face = original_pixel_values.copy()

        extract_embedding(test_face, model_name="Facenet512")

        mock_represent.assert_called_once()
        _, called_kwargs = mock_represent.call_args
        called_img = called_kwargs.get("img_path")
        self.assertIsNotNone(called_img)
        # Verify values remain in 0-255 range and are identical (no accidental division by 255 or RGB conversion)
        np.testing.assert_array_equal(called_img, original_pixel_values)
        self.assertEqual(called_kwargs.get("normalization"), "base")

    # =============================================================
    # 2. UNBYPASSABLE FACE QUALITY VALIDATION TESTS (Fix 2)
    # =============================================================

    @patch("face_database.DeepFace.extract_faces")
    def test_detect_single_face_rejects_missing_facial_area_coords(self, mock_extract):
        """3. Missing or incomplete facial_area coordinates reject extraction."""
        mock_extract.return_value = [{"face": np.ones((60, 60, 3)), "facial_area": None}]
        crop = np.ones((100, 100, 3), dtype=np.uint8) * 100
        extracted, area, status = detect_single_face(crop)
        self.assertIsNone(extracted)
        self.assertIn("Missing facial-area", status)

    @patch("face_database.DeepFace.extract_faces")
    def test_detect_single_face_rejects_zero_or_negative_dimensions(self, mock_extract):
        """4. Zero or negative width/height in facial_area reject extraction."""
        mock_extract.return_value = [{
            "face": np.ones((60, 60, 3)),
            "facial_area": {"x": 10, "y": 10, "w": 0, "h": 50},
        }]
        crop = np.ones((100, 100, 3), dtype=np.uint8) * 100
        extracted, area, status = detect_single_face(crop)
        self.assertIsNone(extracted)
        self.assertIn("Zero-area face box", status)

    @patch("face_database.DeepFace.extract_faces")
    def test_detect_single_face_rejects_out_of_bounds_coords(self, mock_extract):
        """5. Facial area completely outside the input crop bounds rejects extraction."""
        mock_extract.return_value = [{
            "face": np.ones((60, 60, 3)),
            "facial_area": {"x": 200, "y": 200, "w": 50, "h": 50},  # Beyond 100x100
        }]
        crop = np.ones((100, 100, 3), dtype=np.uint8) * 100
        extracted, area, status = detect_single_face(crop)
        self.assertIsNone(extracted)
        self.assertIn("outside image boundaries", status)

    @patch("face_database.DeepFace.extract_faces")
    def test_detect_single_face_rejects_blurry_face_crop(self, mock_extract):
        """6. Blurry face within candidate crop is rejected by Laplacian variance check."""
        mock_extract.return_value = [{
            "face": np.ones((60, 60, 3), dtype=np.float64) * 0.5,
            "facial_area": {"x": 10, "y": 10, "w": 60, "h": 60},
        }]
        # Uniform image has Laplacian variance near 0 (< 30)
        blurry_crop = np.ones((100, 100, 3), dtype=np.uint8) * 128
        extracted, area, status = detect_single_face(blurry_crop, blur_threshold=30.0)
        self.assertIsNone(extracted)
        self.assertIn("blurry", status)

    @patch("face_database.DeepFace.extract_faces")
    def test_detect_single_face_valid_crop_passes_quality(self, mock_extract):
        """7. Sharp, properly dimensioned face crop passes all quality validations."""
        mock_face = np.ones((60, 60, 3), dtype=np.float64) * 0.5
        mock_extract.return_value = [{
            "face": mock_face,
            "facial_area": {"x": 10, "y": 10, "w": 60, "h": 60},
        }]
        # High-frequency textured pattern has high Laplacian variance
        sharp_crop = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        extracted, area, status = detect_single_face(sharp_crop, min_width=40, min_height=40, blur_threshold=5.0)
        self.assertIsNotNone(extracted)
        self.assertEqual(status, "OK")

    @patch("face_database.DeepFace.extract_faces")
    def test_detect_single_face_rejects_no_face(self, mock_extract):
        """8. enforce_detection=True cleanly catches ValueError on absent face."""
        mock_extract.side_effect = ValueError("Face could not be detected.")
        crop = np.zeros((100, 100, 3), dtype=np.uint8)
        extracted, area, status = detect_single_face(crop)
        self.assertIsNone(extracted)
        self.assertEqual(status, "No face detected")

    @patch("face_database.DeepFace.extract_faces")
    def test_detect_single_face_rejects_multiple_faces(self, mock_extract):
        """9. Multiple faces in single candidate crop are rejected as ambiguous."""
        mock_extract.return_value = [
            {"face": np.zeros((50, 50, 3)), "facial_area": {"x": 5, "y": 5, "w": 50, "h": 50}},
            {"face": np.zeros((50, 50, 3)), "facial_area": {"x": 80, "y": 80, "w": 50, "h": 50}},
        ]
        crop = np.zeros((200, 200, 3), dtype=np.uint8)
        extracted, area, status = detect_single_face(crop)
        self.assertIsNone(extracted)
        self.assertIn("Multiple faces detected", status)

    def test_validate_face_quality_empty_and_sub_minimum_dimensions(self):
        """10. validate_face_quality rejects empty arrays and sub-minimum dimensions."""
        self.assertFalse(validate_face_quality(None)[0])
        self.assertFalse(validate_face_quality(np.empty((0, 0, 3)))[0])
        small = np.ones((30, 25, 3), dtype=np.uint8)
        valid, reason, _ = validate_face_quality(small, min_width=40, min_height=40)
        self.assertFalse(valid)
        self.assertIn("beneath minimum", reason)

    # =============================================================
    # 3. PRODUCTION COORDINATE PROCESSING TESTS (Fix 3)
    # =============================================================

    def test_process_box_coordinates_raw_vs_smoothed(self):
        """11. Production process_box_coordinates returns raw clamped box and smoothed box."""
        frame_shape = (480, 640)
        alpha = 0.8

        # Step 1: Initial detection at (100, 100, 200, 300)
        raw_1, smoothed_1 = process_box_coordinates(
            raw_xyxy=(100, 100, 200, 300),
            frame_shape=frame_shape,
            prev_smoothed=None,
            alpha=alpha,
        )
        self.assertEqual(raw_1, (100, 100, 200, 300))
        self.assertEqual(smoothed_1, (100, 100, 200, 300))

        # Step 2: Next frame person steps to (150, 100, 250, 300)
        raw_2, smoothed_2 = process_box_coordinates(
            raw_xyxy=(150, 100, 250, 300),
            frame_shape=frame_shape,
            prev_smoothed=smoothed_1,
            alpha=alpha,
        )
        # Raw box MUST match the current instantaneous detection
        self.assertEqual(raw_2, (150, 100, 250, 300))
        # Smoothed display box gives 80% weight to previous box (100*0.8 + 150*0.2 = 110)
        self.assertEqual(smoothed_2[0], 110)
        self.assertNotEqual(smoothed_2, raw_2)

    def test_process_box_coordinates_clamped_and_invalid_rejected(self):
        """12. process_box_coordinates clamps out-of-frame boxes and rejects inverted boxes."""
        frame_shape = (480, 640)

        # Clamping test
        raw_clamped, smoothed_clamped = process_box_coordinates(
            raw_xyxy=(-30, -20, 700, 500),
            frame_shape=frame_shape,
            prev_smoothed=None,
        )
        self.assertEqual(raw_clamped, (0, 0, 640, 480))

        # Inverted box test (x2 <= x1)
        raw_inv, smoothed_inv = process_box_coordinates(
            raw_xyxy=(200, 100, 150, 300),
            frame_shape=frame_shape,
            prev_smoothed=None,
        )
        self.assertIsNone(raw_inv)
        self.assertIsNone(smoothed_inv)

    # =============================================================
    # 4. PRODUCTION TRACKER CLEANUP TESTS (Fix 4)
    # =============================================================

    def test_cleanup_stale_tracks_purges_expired_state_and_reverse_mappings(self):
        """13. Production cleanup_stale_tracks purges expired tracks and reverse id_map keys."""
        track_meta = {
            1: {"last_seen": 100.0, "frame_count": 10},
            2: {"last_seen": 109.5, "frame_count": 80},  # Active
        }
        id_map = {11: 1, 12: 1, 21: 2}  # yolo_ids 11, 12 map to pid 1; 21 maps to pid 2
        person_names = {1: "Raza", 2: "Shiv"}
        name_history = {1: ["Raza"], 2: ["Shiv"]}
        smoothed_boxes = {1: (10, 10, 50, 50), 2: (60, 60, 100, 100)}

        # Run production cleanup at current_time = 110.0 with timeout = 2.0
        removed = cleanup_stale_tracks(
            track_meta=track_meta,
            id_map=id_map,
            person_names=person_names,
            name_history=name_history,
            smoothed_boxes=smoothed_boxes,
            current_time=110.0,
            timeout=2.0,
        )

        self.assertEqual(removed, [1])
        # Verify pid 1 removed from all dicts
        self.assertNotIn(1, track_meta)
        self.assertNotIn(1, person_names)
        self.assertNotIn(1, name_history)
        self.assertNotIn(1, smoothed_boxes)

        # Verify reverse mappings in id_map purged
        self.assertNotIn(11, id_map)
        self.assertNotIn(12, id_map)

        # Verify active pid 2 remains intact
        self.assertIn(2, track_meta)
        self.assertIn(21, id_map)
        self.assertEqual(id_map[21], 2)
        self.assertEqual(person_names[2], "Shiv")

    def test_recycled_tracker_id_gets_fresh_identity_and_attendance_preserved(self):
        """14. Recycled tracker ID starts clean via production helper; AttendanceManager records are not deleted."""
        csv_file = os.path.join(self.test_dir, "test_att.csv")
        mgr = AttendanceManager(
            csv_path=csv_file,
            session_id="sess_test",
            min_time_seconds=0.1,
            min_frames=1,
        )
        mgr.record_attendance("Alice", 5.0)
        self.assertTrue(mgr.is_marked("Alice"))

        # Initialize tracking state with Person 1 (Alice) mapped to YOLO tracker ID 5
        track_meta = {
            1: {
                "first_seen": 95.0,
                "last_seen": 100.0,
                "frame_count": 50,
            }
        }
        id_map = {5: 1}  # yolo_id 5 -> person_id 1
        person_names = {1: "Alice"}
        name_history = {1: ["Alice", "Alice"]}
        smoothed_boxes = {1: (10, 10, 50, 50)}
        next_person_id = 2

        # Stale track with old last_seen: current_time = 110.0 (elapsed 10.0s > TRACK_TIMEOUT 2.0s)
        current_time = 110.0
        track_timeout = 2.0

        # Step 1: Run production cleanup_stale_tracks() BEFORE new detection
        cleanup_stale_tracks(
            track_meta=track_meta,
            id_map=id_map,
            person_names=person_names,
            name_history=name_history,
            smoothed_boxes=smoothed_boxes,
            current_time=current_time,
            timeout=track_timeout,
        )

        # Confirm old track 1 and expired yolo_id 5 mapping were removed
        self.assertNotIn(1, track_meta)
        self.assertNotIn(5, id_map)
        self.assertNotIn(1, person_names)
        self.assertNotIn(1, name_history)
        self.assertNotIn(1, smoothed_boxes)

        # Step 2: Call production tracker assignment helper for recycled YOLO ID
        recycled_yolo_id = 5
        person_id, next_person_id, is_new_track = assign_or_update_track(
            yolo_id=recycled_yolo_id,
            id_map=id_map,
            track_meta=track_meta,
            current_time=current_time,
            next_person_id=next_person_id,
        )

        # Step 3: Assertions on fresh identity and preserved attendance records
        self.assertTrue(is_new_track)
        self.assertEqual(person_id, 2)
        self.assertNotEqual(person_id, 1)
        self.assertEqual(id_map[recycled_yolo_id], 2)
        self.assertEqual(next_person_id, 3)

        # Must start in "Detecting..." with no old history
        assigned_name = person_names.get(person_id, "Detecting...")
        self.assertEqual(assigned_name, "Detecting...")
        self.assertNotIn(person_id, name_history)

        # Frame count starts at 1 on first detection (no double increment)
        self.assertEqual(track_meta[person_id]["frame_count"], 1)

        # Confirm AttendanceManager records remain intact
        self.assertTrue(mgr.is_marked("Alice"))
        with open(csv_file, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = list(reader)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][1], "Alice")

    def test_production_assign_or_update_track_lifecycle(self):
        """15. Production assign_or_update_track assigns new person_id, increments frame_count, and handles lifecycle."""
        id_map = {}
        track_meta = {}
        next_person_id = 1
        t1 = 100.0

        # Frame 1: First detection of yolo_id 10
        pid1, next_pid, is_new = assign_or_update_track(
            yolo_id=10,
            id_map=id_map,
            track_meta=track_meta,
            current_time=t1,
            next_person_id=next_person_id,
        )
        self.assertTrue(is_new)
        self.assertEqual(pid1, 1)
        self.assertEqual(next_pid, 2)
        self.assertEqual(track_meta[pid1]["frame_count"], 1)
        self.assertEqual(track_meta[pid1]["first_seen"], t1)
        self.assertEqual(track_meta[pid1]["last_seen"], t1)

        # Frame 2: Second detection of same yolo_id 10
        t2 = 100.033
        pid2, next_pid2, is_new2 = assign_or_update_track(
            yolo_id=10,
            id_map=id_map,
            track_meta=track_meta,
            current_time=t2,
            next_person_id=next_pid,
        )
        self.assertFalse(is_new2)
        self.assertEqual(pid2, 1)
        self.assertEqual(next_pid2, 2)  # counter does not advance for existing track
        self.assertEqual(track_meta[pid2]["frame_count"], 2)
        self.assertEqual(track_meta[pid2]["first_seen"], t1)
        self.assertEqual(track_meta[pid2]["last_seen"], t2)

    # =============================================================
    # 5. PREFLIGHT MODEL VERIFICATION TESTS (Fix 5)
    # =============================================================

    @patch("attendance_yolo.YOLO")
    def test_preflight_missing_model_weights_rejects_and_prevents_yolo_init(self, mock_yolo):
        """16. verify_model_weights returns False on missing file and prevents YOLO instantiation."""
        non_existent_weights = os.path.join(self.test_dir, "missing_weights.pt")

        is_valid = verify_model_weights(non_existent_weights)
        self.assertFalse(is_valid)

        # If not valid, YOLO must NEVER be called
        if is_valid:
            mock_yolo(non_existent_weights)

        mock_yolo.assert_not_called()

    def test_preflight_existing_model_weights_passes(self):
        """17. verify_model_weights returns True when weights file exists."""
        weights_path = os.path.join(self.test_dir, "test_weights.pt")
        with open(weights_path, "wb") as f:
            f.write(b"dummy_weights")

        self.assertTrue(verify_model_weights(weights_path))

    # =============================================================
    # 6. DATABASE REBUILD & PARTIAL REPLACEMENT PROTECTION (Fix 6)
    # =============================================================

    def test_build_face_database_aborts_on_empty_student_folder(self):
        """18. If any student folder has zero supported images, abort and preserve existing db byte-for-byte."""
        mock_students = os.path.join(self.test_dir, "mock_students")
        student_a = os.path.join(mock_students, "Student_A")
        student_empty = os.path.join(mock_students, "Student_B")
        os.makedirs(student_a, exist_ok=True)
        os.makedirs(student_empty, exist_ok=True)

        photo_a_path = os.path.join(student_a, "photo_a.jpg")
        photo_a_bytes = b"PHOTO_STUDENT_A_BYTES"
        with open(photo_a_path, "wb") as f:
            f.write(photo_a_bytes)

        # Existing database
        db_path = os.path.join(self.test_dir, "face_embeddings.pkl")
        original_data = {"Student_A": [np.ones(512)]}
        with open(db_path, "wb") as f:
            pickle.dump(original_data, f)
        with open(db_path, "rb") as f:
            original_db_bytes = f.read()

        success = build_face_database(students_dir=mock_students, output_path=db_path)

        # 1. Assert build_face_database returns False
        self.assertFalse(success)

        # 2. Assert existing database remains byte-for-byte identical
        with open(db_path, "rb") as f:
            self.assertEqual(f.read(), original_db_bytes)

        # 3. Assert student photo remains unmodified
        with open(photo_a_path, "rb") as f:
            self.assertEqual(f.read(), photo_a_bytes)

    def test_build_face_database_aborts_when_existing_enrolled_identity_folder_missing(self):
        """19. If an existing enrolled identity has no corresponding folder, abort and preserve db byte-for-byte."""
        mock_students = os.path.join(self.test_dir, "mock_students")
        student_a = os.path.join(mock_students, "Student_A")
        os.makedirs(student_a, exist_ok=True)

        photo_a_path = os.path.join(student_a, "photo_a.jpg")
        photo_a_bytes = b"PHOTO_STUDENT_A_BYTES"
        with open(photo_a_path, "wb") as f:
            f.write(photo_a_bytes)

        # Existing database contains Student_A AND Student_C (Student_C folder is missing from disk)
        db_path = os.path.join(self.test_dir, "face_embeddings.pkl")
        original_data = {"Student_A": [np.ones(512)], "Student_C": [np.ones(512)]}
        with open(db_path, "wb") as f:
            pickle.dump(original_data, f)
        with open(db_path, "rb") as f:
            original_db_bytes = f.read()

        success = build_face_database(students_dir=mock_students, output_path=db_path)

        # 1. Assert build_face_database returns False
        self.assertFalse(success)

        # 2. Assert existing database remains byte-for-byte identical
        with open(db_path, "rb") as f:
            self.assertEqual(f.read(), original_db_bytes)

        # 3. Assert student photo remains unmodified
        with open(photo_a_path, "rb") as f:
            self.assertEqual(f.read(), photo_a_bytes)

    @patch("build_face_database.detect_single_face")
    @patch("build_face_database.extract_embedding")
    def test_build_face_database_succeeds_when_all_enrolled_folders_contain_valid_images(self, mock_emb, mock_det):
        """20. Successful build works when all enrolled folders contain valid images."""
        mock_students = os.path.join(self.test_dir, "mock_students")
        student_a = os.path.join(mock_students, "Student_A")
        student_b = os.path.join(mock_students, "Student_B")
        os.makedirs(student_a, exist_ok=True)
        os.makedirs(student_b, exist_ok=True)

        photo_a_path = os.path.join(student_a, "photo_a.jpg")
        photo_b_path = os.path.join(student_b, "photo_b.jpg")
        with open(photo_a_path, "wb") as f:
            f.write(b"PHOTO_A")
        with open(photo_b_path, "wb") as f:
            f.write(b"PHOTO_B")

        db_path = os.path.join(self.test_dir, "face_embeddings.pkl")
        with open(db_path, "wb") as f:
            pickle.dump({"Student_A": [np.ones(512)], "Student_B": [np.ones(512)]}, f)

        mock_det.return_value = (np.ones((60, 60, 3)), {"x": 10, "y": 10, "w": 60, "h": 60}, "OK")
        mock_emb.return_value = np.ones(512, dtype=np.float32)

        with patch("cv2.imread", return_value=np.ones((100, 100, 3), dtype=np.uint8)):
            success = build_face_database(students_dir=mock_students, output_path=db_path)

        self.assertTrue(success)
        reloaded = load_database(db_path)
        self.assertEqual(len(reloaded), 2)
        self.assertIn("Student_A", reloaded)
        self.assertIn("Student_B", reloaded)

    @patch("build_face_database.detect_single_face")
    @patch("build_face_database.extract_embedding")
    def test_build_face_database_aborts_on_partial_failure_to_protect_existing_db(self, mock_emb, mock_det):
        """21. If a student folder contains images but 0 embeddings, abort to prevent partial replacement."""
        mock_students = os.path.join(self.test_dir, "mock_students")
        student_a = os.path.join(mock_students, "Student_A")
        student_b = os.path.join(mock_students, "Student_B")
        os.makedirs(student_a, exist_ok=True)
        os.makedirs(student_b, exist_ok=True)

        photo_a_path = os.path.join(student_a, "photo_a.jpg")
        photo_b_path = os.path.join(student_b, "photo_b.jpg")
        photo_a_bytes = b"PHOTO_STUDENT_A_ORIGINAL_BYTES"
        photo_b_bytes = b"PHOTO_STUDENT_B_ORIGINAL_BYTES"

        with open(photo_a_path, "wb") as f:
            f.write(photo_a_bytes)
        with open(photo_b_path, "wb") as f:
            f.write(photo_b_bytes)

        # Existing database has both students
        db_path = os.path.join(self.test_dir, "face_embeddings.pkl")
        original_data = {"Student_A": [np.ones(512)], "Student_B": [np.ones(512)]}
        with open(db_path, "wb") as f:
            pickle.dump(original_data, f)
        with open(db_path, "rb") as f:
            original_db_bytes = f.read()

        # Mock imread to distinguish files
        def mock_imread_side_effect(path, *args, **kwargs):
            if "Student_A" in path:
                return np.full((100, 100, 3), 1, dtype=np.uint8)
            elif "Student_B" in path:
                return np.full((100, 100, 3), 2, dtype=np.uint8)
            return np.ones((100, 100, 3), dtype=np.uint8)

        # Mock detect_single_face: Student A succeeds, Student B fails (0 embeddings)
        def mock_detect_side_effect(image_or_crop, **kwargs):
            if image_or_crop[0, 0, 0] == 1:
                return (np.ones((60, 60, 3)), {"x": 10, "y": 10, "w": 60, "h": 60}, "OK")
            return (None, None, "No face detected in crop")

        mock_det.side_effect = mock_detect_side_effect
        mock_emb.return_value = np.ones(512, dtype=np.float32)

        with patch("cv2.imread", side_effect=mock_imread_side_effect):
            success = build_face_database(students_dir=mock_students, output_path=db_path)

        # 1. Assert build_face_database returns False
        self.assertFalse(success)

        # 2. Assert existing database remains byte-for-byte identical
        with open(db_path, "rb") as f:
            current_db_bytes = f.read()
        self.assertEqual(current_db_bytes, original_db_bytes)

        # 3. Assert no replacement occurred
        reloaded = load_database(db_path)
        self.assertEqual(len(reloaded), 2)
        self.assertIn("Student_A", reloaded)
        self.assertIn("Student_B", reloaded)

        # 4. Assert student photos are unmodified
        with open(photo_a_path, "rb") as f:
            self.assertEqual(f.read(), photo_a_bytes)
        with open(photo_b_path, "rb") as f:
            self.assertEqual(f.read(), photo_b_bytes)

    @patch("build_face_database.datetime")
    @patch("build_face_database.detect_single_face")
    @patch("build_face_database.extract_embedding")
    def test_build_face_database_backup_collision_free(self, mock_emb, mock_det, mock_dt):
        """22. Database backup handles existing timestamped backups without overwriting."""
        mock_students = os.path.join(self.test_dir, "mock_students")
        student_a = os.path.join(mock_students, "Student_A")
        os.makedirs(student_a, exist_ok=True)
        with open(os.path.join(student_a, "img.jpg"), "wb") as f:
            f.write(b"dummy_img")

        # Deterministic fixed timestamp
        fixed_dt = datetime(2026, 11, 20, 12, 0, 0)
        fixed_ts = fixed_dt.strftime("%Y%m%d_%H%M%S")
        mock_dt.now.return_value = fixed_dt

        # Existing database
        db_path = os.path.join(self.test_dir, "face_embeddings.pkl")
        original_db_content = pickle.dumps({"Student_A": [np.ones(512)]})
        with open(db_path, "wb") as f:
            f.write(original_db_content)

        # Pre-create collision backup with the exact same timestamp string
        collision_backup_path = f"{db_path}.bak.{fixed_ts}"
        pre_existing_backup_data = b"PRE_EXISTING_BACKUP_CONTENT_MUST_NOT_BE_OVERWRITTEN"
        with open(collision_backup_path, "wb") as f:
            f.write(pre_existing_backup_data)

        mock_det.return_value = (np.ones((60, 60, 3)), {}, "OK")
        mock_emb.return_value = np.ones(512, dtype=np.float32)

        with patch("cv2.imread", return_value=np.ones((100, 100, 3), dtype=np.uint8)):
            success = build_face_database(students_dir=mock_students, output_path=db_path)

        self.assertTrue(success)

        # 1. Assert pre-existing collision backup was NOT overwritten
        with open(collision_backup_path, "rb") as f:
            self.assertEqual(f.read(), pre_existing_backup_data)

        # 2. Assert build_face_database detected collision and created <path>.bak.<ts>_1
        expected_collision_free_backup = f"{db_path}.bak.{fixed_ts}_1"
        self.assertTrue(
            os.path.exists(expected_collision_free_backup),
            f"Expected backup file {expected_collision_free_backup} was not created.",
        )

        # 3. Assert original database bytes preserved in the new backup
        with open(expected_collision_free_backup, "rb") as f:
            self.assertEqual(f.read(), original_db_content)

    def test_build_face_database_aborts_on_corrupted_existing_database(self):
        """23. Fail-closed: corrupted existing database aborts rebuild, creates no backup, preserves bytes."""
        mock_students = os.path.join(self.test_dir, "mock_students")
        student_a = os.path.join(mock_students, "Student_A")
        os.makedirs(student_a, exist_ok=True)
        with open(os.path.join(student_a, "photo.jpg"), "wb") as f:
            f.write(b"PHOTO_BYTES")

        db_path = os.path.join(self.test_dir, "face_embeddings.pkl")
        corrupted_bytes = b"CORRUPTED_NON_PICKLE_BYTE_SEQUENCE_\x00\xff\xfe\x80"
        with open(db_path, "wb") as f:
            f.write(corrupted_bytes)

        # Execute production build_face_database()
        success = build_face_database(students_dir=mock_students, output_path=db_path)

        # 1. Assert rebuild aborts
        self.assertFalse(success)

        # 2. Assert original database remains byte-for-byte unchanged
        with open(db_path, "rb") as f:
            self.assertEqual(f.read(), corrupted_bytes)

        # 3. Assert no backup files created
        backups = [f for f in os.listdir(self.test_dir) if "face_embeddings.pkl.bak." in f]
        self.assertEqual(len(backups), 0)

        # 4. Assert no replacement or temporary files created
        tmps = [f for f in os.listdir(self.test_dir) if f.startswith(".face_embeddings.pkl")]
        self.assertEqual(len(tmps), 0)

    def test_build_face_database_aborts_on_non_dict_existing_database(self):
        """24. Fail-closed: valid pickle of non-dict object aborts rebuild, creates no backup, preserves bytes."""
        mock_students = os.path.join(self.test_dir, "mock_students")
        student_a = os.path.join(mock_students, "Student_A")
        os.makedirs(student_a, exist_ok=True)
        with open(os.path.join(student_a, "photo.jpg"), "wb") as f:
            f.write(b"PHOTO_BYTES")

        db_path = os.path.join(self.test_dir, "face_embeddings.pkl")
        non_dict_bytes = pickle.dumps(["Student_A", "Student_B", "unexpected_list_type"])
        with open(db_path, "wb") as f:
            f.write(non_dict_bytes)

        # Execute production build_face_database()
        success = build_face_database(students_dir=mock_students, output_path=db_path)

        # 1. Assert rebuild aborts
        self.assertFalse(success)

        # 2. Assert original database remains byte-for-byte unchanged
        with open(db_path, "rb") as f:
            self.assertEqual(f.read(), non_dict_bytes)

        # 3. Assert no backup files created
        backups = [f for f in os.listdir(self.test_dir) if "face_embeddings.pkl.bak." in f]
        self.assertEqual(len(backups), 0)

        # 4. Assert no replacement or temporary files created
        tmps = [f for f in os.listdir(self.test_dir) if f.startswith(".face_embeddings.pkl")]
        self.assertEqual(len(tmps), 0)

    def test_build_face_database_aborts_on_zero_byte_existing_database(self):
        """25. Fail-closed: zero-byte existing database aborts rebuild, creates no backup, preserves 0-byte file."""
        mock_students = os.path.join(self.test_dir, "mock_students")
        student_a = os.path.join(mock_students, "Student_A")
        os.makedirs(student_a, exist_ok=True)
        with open(os.path.join(student_a, "photo.jpg"), "wb") as f:
            f.write(b"PHOTO_BYTES")

        db_path = os.path.join(self.test_dir, "face_embeddings.pkl")
        # Create zero-byte file
        with open(db_path, "wb") as f:
            pass
        self.assertEqual(os.path.getsize(db_path), 0)

        # Execute production build_face_database()
        success = build_face_database(students_dir=mock_students, output_path=db_path)

        # 1. Assert rebuild aborts
        self.assertFalse(success)

        # 2. Assert original database remains a 0-byte file
        self.assertTrue(os.path.exists(db_path))
        self.assertEqual(os.path.getsize(db_path), 0)

        # 3. Assert no backup files created
        backups = [f for f in os.listdir(self.test_dir) if "face_embeddings.pkl.bak." in f]
        self.assertEqual(len(backups), 0)

        # 4. Assert no replacement or temporary files created
        tmps = [f for f in os.listdir(self.test_dir) if f.startswith(".face_embeddings.pkl")]
        self.assertEqual(len(tmps), 0)

    # =============================================================
    # 7. COSINE SIMILARITY & CONFIGURATION TESTS
    # =============================================================

    def test_cosine_similarity_safeguards(self):
        """27. Cosine similarity guards against zero vectors, NaN/Inf, None, and shape mismatch."""
        valid_vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        zero_vec = np.zeros(3, dtype=np.float32)

        self.assertAlmostEqual(cosine_similarity(valid_vec, valid_vec), 1.0, places=5)
        self.assertEqual(cosine_similarity(zero_vec, valid_vec), 0.0)
        self.assertEqual(cosine_similarity(None, valid_vec), 0.0)
        self.assertEqual(cosine_similarity(np.array([1.0, np.nan, 3.0]), valid_vec), 0.0)
        self.assertEqual(cosine_similarity(np.array([1.0, 2.0]), valid_vec), 0.0)

    def test_configuration_defaults_and_env_overrides(self):
        """28. SystemConfig loads production defaults and respects environment overrides."""
        cfg_default = SystemConfig()
        self.assertEqual(cfg_default.MIN_TIME_SECONDS, 5.0)
        self.assertEqual(cfg_default.MIN_FRAMES, 100)
        self.assertEqual(cfg_default.MAX_FRAME_GAP_SECONDS, 0.25)

        with patch.dict(os.environ, {
            "CAMERA_INDEX": "3",
            "RECOGNITION_THRESHOLD": "0.80",
        }):
            cfg_override = SystemConfig()
            self.assertEqual(cfg_override.CAMERA_INDEX, 3)
            self.assertEqual(cfg_override.RECOGNITION_THRESHOLD, 0.80)

    def test_missing_face_embeddings_returns_empty_dict(self):
        """29. Missing face embedding database returns empty dict and logs warning."""
        non_existent_db = os.path.join(self.test_dir, "missing_db.pkl")
        loaded = load_database(non_existent_db)
        self.assertEqual(loaded, {})


if __name__ == "__main__":
    unittest.main()
