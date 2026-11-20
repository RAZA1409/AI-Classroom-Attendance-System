"""
Face Database Builder Module
============================
Pre-computes and caches 512-dimensional facial embeddings for all enrolled
students in the 'students/' directory.

Guarantees 100% pipeline compatibility with live recognition by using
identical face detection, BGR standardization, quality validation, and FaceNet512 embedding.

Safe Rebuild Guarantees:
- Never deletes or modifies student photographs.
- If any enrolled student folder fails to produce valid embeddings, aborts replacement
  to prevent partial database corruption.
- Automatically creates a collision-free timestamped backup of any existing database.
- Writes atomically via a temporary file.
- Does not log student names, photographs, or biometric vectors.
"""

import os
import shutil
import pickle
from datetime import datetime
import cv2

from config import config
from logger import get_logger
from face_database import detect_single_face, extract_embedding

logger = get_logger("build_face_database")


def build_face_database(
    students_dir: str = config.STUDENTS_DIR,
    output_path: str = config.DATABASE_PATH,
    detector_backend: str = config.FACE_DETECTOR_BACKEND,
    model_name: str = config.FACE_RECOGNITION_MODEL,
) -> bool:
    """
    Build facial embedding database with partial-replacement protection,
    collision-free backup, and identical pipeline preprocessing.
    """
    if not os.path.exists(students_dir):
        logger.error("Students directory not found. Please create it and enroll students first.")
        return False

    logger.info("Starting face database build from '%s' using detector '%s'...", students_dir, detector_backend)
    database = {}
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    student_entries = [d for d in os.listdir(students_dir) if os.path.isdir(os.path.join(students_dir, d))]
    if not student_entries:
        logger.warning("No student folders found inside '%s'.", students_dir)
        return False

    # Guard 1: Verify existing database integrity and enrolled identities (fail-closed)
    if os.path.exists(output_path):
        if os.path.getsize(output_path) == 0:
            logger.error(
                "Build aborted: Existing database at '%s' is zero-byte (empty/corrupted). "
                "Existing database preserved to prevent unintended data loss.",
                output_path,
            )
            print(
                f"❌ Error: Existing face database '{output_path}' is a zero-byte file. "
                f"Database rebuild aborted to protect existing data."
            )
            return False

        try:
            with open(output_path, "rb") as f:
                existing_db = pickle.load(f)
        except Exception as e:
            logger.error(
                "Build aborted: Existing database at '%s' could not be loaded or is corrupted: %s. "
                "Existing database preserved to prevent unintended data loss.",
                output_path,
                e,
            )
            print(
                f"❌ Error: Existing face database '{output_path}' is corrupted or unreadable. "
                f"Database rebuild aborted to protect existing data."
            )
            return False

        if not isinstance(existing_db, dict):
            logger.error(
                "Build aborted: Existing database at '%s' is not a valid dictionary object (found %s). "
                "Existing database preserved to prevent unintended data loss.",
                output_path,
                type(existing_db).__name__,
            )
            print(
                f"❌ Error: Existing face database '{output_path}' is not a valid dictionary. "
                f"Database rebuild aborted to protect existing data."
            )
            return False

        existing_identities = set(existing_db.keys())
        discovered_folders = set(student_entries)
        missing_folders = existing_identities - discovered_folders
        if missing_folders:
            logger.error(
                "Build aborted: %d existing enrolled identity/identities have no corresponding folder in '%s'. "
                "Existing database preserved to prevent unintended identity deletion.",
                len(missing_folders),
                students_dir,
            )
            print(
                f"❌ Error: {len(missing_folders)} existing enrolled identity/identities have no corresponding folder. "
                f"Database replacement aborted to protect existing records."
            )
            return False

    # Guard 2: Verify that every discovered student folder contains supported image files
    empty_student_folders_count = 0
    for student in student_entries:
        student_folder = os.path.join(students_dir, student)
        image_files = [f for f in os.listdir(student_folder) if f.lower().endswith(valid_exts)]
        if not image_files:
            empty_student_folders_count += 1

    if empty_student_folders_count > 0:
        logger.error(
            "Build aborted: %d student folder(s) contain zero supported image files. "
            "Existing database preserved to prevent partial replacement.",
            empty_student_folders_count,
        )
        print(
            f"❌ Error: {empty_student_folders_count} student folder(s) contain zero supported image files. "
            f"Database replacement aborted to protect existing records."
        )
        return False

    failed_student_folders_count = 0
    total_folders = len(student_entries)

    for idx, student in enumerate(student_entries):
        student_folder = os.path.join(students_dir, student)
        embeddings = []
        image_files = [f for f in os.listdir(student_folder) if f.lower().endswith(valid_exts)]

        for file in image_files:
            img_path = os.path.join(student_folder, file)
            img = cv2.imread(img_path)

            if img is None:
                logger.warning("Could not read image in student folder %d. Skipping file.", idx + 1)
                continue

            # 1. Genuine face detection and quality check (matches live pipeline with BGR standardization)
            face_arr, _, status = detect_single_face(
                image_or_crop=img,
                detector_backend=detector_backend,
            )
            if face_arr is None:
                logger.debug("Rejected image in folder %d: %s", idx + 1, status)
                continue

            # 2. Extract FaceNet512 embedding from aligned face crop
            emb = extract_embedding(face_arr, model_name=model_name)
            if emb is not None:
                embeddings.append(emb)
            else:
                logger.debug("Failed to compute embedding for image in folder %d.", idx + 1)

        # Partial Replacement Protection:
        # If folder had image files but zero valid embeddings were produced, flag as failure
        if embeddings:
            database[student] = embeddings
            logger.info("Processed student folder %d of %d (%d valid embeddings extracted).", idx + 1, total_folders, len(embeddings))
        else:
            failed_student_folders_count += 1
            logger.warning("Student folder %d of %d produced zero valid face embeddings.", idx + 1, total_folders)

    # If any student folder with images failed, abort to protect existing complete database
    if failed_student_folders_count > 0:
        logger.error(
            "Build aborted: %d student folder(s) contain images but produced zero valid face embeddings. "
            "Existing database preserved to prevent partial replacement.",
            failed_student_folders_count,
        )
        print(
            f"❌ Error: {failed_student_folders_count} student folder(s) produced zero valid embeddings. "
            f"Database replacement aborted to protect existing records."
        )
        return False

    if not database:
        logger.warning("No valid student embeddings found. Database will NOT be overwritten with empty data.")
        return False

    # Safe collision-free backup of existing database before replacement
    if os.path.exists(output_path):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{output_path}.bak.{timestamp}"
        counter = 0
        while os.path.exists(backup_path):
            counter += 1
            backup_path = f"{output_path}.bak.{timestamp}_{counter}"

        try:
            shutil.copy2(output_path, backup_path)
            logger.info("Created safe database backup at '%s'", backup_path)
        except Exception as e:
            logger.error("Failed to create backup of existing database: %s. Aborting write.", e)
            return False

    # Atomic write using temporary file in same directory
    dir_name = os.path.dirname(os.path.abspath(output_path))
    temp_output = os.path.join(dir_name, f".{os.path.basename(output_path)}.tmp")
    try:
        with open(temp_output, "wb") as f:
            pickle.dump(database, f)
            f.flush()
        os.replace(temp_output, output_path)
        logger.info("Successfully updated face database: %d total identities enrolled.", len(database))
        return True
    except Exception as e:
        logger.error("Failed to write face database atomically: %s", e)
        if os.path.exists(temp_output):
            try:
                os.remove(temp_output)
            except OSError:
                pass
        return False


if __name__ == "__main__":
    build_face_database()
