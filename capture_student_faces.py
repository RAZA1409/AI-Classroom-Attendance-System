import cv2
import os
import re
from typing import Tuple
from attendance_manager import normalize_student_name


def validate_and_sanitize_student_name(raw_name: str) -> str:
    """
    Validate and sanitize student name for directory creation.
    Requires at least one alphanumeric character.
    Replaces disallowed characters with underscores and strips leading/trailing underscores.
    Returns empty string if invalid.
    """
    name = raw_name.strip()
    if not name:
        return ""

    # Must contain at least one alphanumeric character (rejects symbol-only inputs like '///', '___')
    if not re.search(r'[a-zA-Z0-9]', name):
        return ""

    # Sanitize characters: allow alphanumeric, underscores, hyphens
    clean = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
    # Collapse multiple consecutive underscores
    clean = re.sub(r'_+', '_', clean).strip('_')
    return clean


def check_existing_student_collision(base_dir: str, clean_name: str) -> Tuple[bool, str]:
    """
    Check if clean_name matches an existing student directory using canonical normalization.
    Returns (exists, canonical_name).
    """
    if not os.path.exists(base_dir):
        return False, clean_name

    norm_target = normalize_student_name(clean_name)
    for entry in os.listdir(base_dir):
        entry_path = os.path.join(base_dir, entry)
        if os.path.isdir(entry_path):
            if normalize_student_name(entry) == norm_target:
                return True, entry
    return False, clean_name


def get_next_image_index(save_path: str, clean_name: str) -> int:
    """
    Determine the next available numerical index for capturing student images,
    ensuring existing photographs are never overwritten.
    """
    if not os.path.exists(save_path):
        return 0

    existing_files = os.listdir(save_path)
    existing_indices = []
    pattern = re.compile(rf"^{re.escape(clean_name)}_(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)

    for f in existing_files:
        match = pattern.match(f)
        if match:
            existing_indices.append(int(match.group(1)))

    if existing_indices:
        return max(existing_indices) + 1
    elif existing_files:
        return len(existing_files)
    return 0


def main():
    raw_name = input("Enter student name: ")
    clean_name = validate_and_sanitize_student_name(raw_name)

    if not clean_name:
        print("❌ Invalid student name. Name must contain at least one alphanumeric character (A-Z, a-z, 0-9).")
        exit(1)

    base_dir = os.path.abspath("students")
    collides, existing_name = check_existing_student_collision(base_dir, clean_name)
    if collides and existing_name != clean_name:
        print(f"⚠️ Notice: Student name '{clean_name}' matches existing student folder '{existing_name}'. Using '{existing_name}'.")
        clean_name = existing_name

    save_path = os.path.abspath(os.path.join(base_dir, clean_name))

    # Strict path traversal prevention
    if not save_path.startswith(base_dir):
        print("❌ Invalid path: path traversal detected.")
        exit(1)

    os.makedirs(save_path, exist_ok=True)

    # Protect against silent file overwrites
    start_index = get_next_image_index(save_path, clean_name)
    if start_index > 0:
        print(f"ℹ️ Found existing photos for '{clean_name}'. Appending captures starting at index {start_index}.")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Camera not accessible. Please check your camera connection or index.")
        exit(1)

    count = start_index
    captured_count = 0

    print(f"Enrolling student: {clean_name}")
    print("Press SPACE to capture image")
    print("Press Q to quit")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("❌ Failed to grab frame from camera.")
            break

        cv2.imshow("Capture Faces", frame)

        key = cv2.waitKey(1)

        if key == 32:  # SPACE key
            img_path = os.path.join(save_path, f"{clean_name}_{count}.jpg")
            cv2.imwrite(img_path, frame)
            print(f"Saved: {img_path}")
            count += 1
            captured_count += 1

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"✅ Enrollment finished for {clean_name}. Captured {captured_count} new images (total: {count}).")


if __name__ == "__main__":
    main()
