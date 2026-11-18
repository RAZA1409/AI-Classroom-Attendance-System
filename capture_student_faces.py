
import cv2
import os
import re

student_name = input("Enter student name: ").strip()

# Sanitize student name to prevent path traversal and invalid filename characters
clean_name = re.sub(r'[^a-zA-Z0-9_-]', '_', student_name)
if not clean_name:
    print("❌ Invalid student name. Please use alphanumeric characters, dashes, or underscores.")
    exit(1)

base_dir = os.path.abspath("students")
save_path = os.path.abspath(os.path.join(base_dir, clean_name))

# Ensure target path stays strictly inside the students directory
if not save_path.startswith(base_dir):
    print("❌ Invalid path: path traversal detected.")
    exit(1)

os.makedirs(save_path, exist_ok=True)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Camera not accessible. Please check your camera connection or index.")
    exit(1)

count = 0

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

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"✅ Enrollment finished for {clean_name}. Captured {count} images.")
