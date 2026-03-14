import os
import cv2
import pickle
from deepface import DeepFace

students_path = "students"
database = {}

print("Building face database...")

for student in os.listdir(students_path):
    student_folder = os.path.join(students_path, student)

    if not os.path.isdir(student_folder):
        continue

    embeddings = []

    for file in os.listdir(student_folder):

        if file.lower().endswith((".jpg", ".jpeg", ".png")):

            img_path = os.path.join(student_folder, file)
            img = cv2.imread(img_path)

            if img is None:
                continue

            try:
                rep = DeepFace.represent(
                    img_path=img,
                    model_name="Facenet512",
                    enforce_detection=False
                )

                embeddings.append(rep[0]["embedding"])

            except:
                continue

    if len(embeddings) > 0:
        database[student] = embeddings

print("Students loaded:", len(database))

with open("face_embeddings.pkl", "wb") as f:
    pickle.dump(database, f)

print("Face database saved → face_embeddings.pkl")