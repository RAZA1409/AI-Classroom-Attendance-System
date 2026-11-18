import os
import cv2
import pickle
from deepface import DeepFace

students_path = "students"
database = {}

if not os.path.exists(students_path):
    print(f"❌ Students directory '{students_path}' not found. Please create it and add student image folders.")
    exit(1)

print("Building face database...")

for student in os.listdir(students_path):
    student_folder = os.path.join(students_path, student)

    if not os.path.isdir(student_folder):
        continue

    embeddings = []

    for file in os.listdir(student_folder):

        if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):

            img_path = os.path.join(student_folder, file)
            img = cv2.imread(img_path)

            if img is None:
                print(f"⚠️ Warning: Could not read image '{img_path}'. Skipping.")
                continue

            try:
                rep = DeepFace.represent(
                    img_path=img,
                    model_name="Facenet512",
                    enforce_detection=False
                )

                embeddings.append(rep[0]["embedding"])

            except Exception as e:
                print(f"⚠️ Warning: Failed to extract embedding for '{img_path}': {e}")
                continue

    if len(embeddings) > 0:
        database[student] = embeddings

print("Students loaded:", len(database))

if not database:
    print("⚠️ No valid student embeddings were found. Database will not be overwritten with empty data.")
    exit(0)

# Atomic write using temporary file to prevent corruption
temp_output = "face_embeddings.pkl.tmp"
with open(temp_output, "wb") as f:
    pickle.dump(database, f)

os.replace(temp_output, "face_embeddings.pkl")
print("Face database saved → face_embeddings.pkl")
