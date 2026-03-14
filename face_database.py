import os
import cv2
import numpy as np
from deepface import DeepFace

# ==============================
# Load student embeddings
# ==============================
def load_student_database(base_path="students"):

    database = {}

    for student_name in os.listdir(base_path):

        student_folder = os.path.join(base_path, student_name)

        if not os.path.isdir(student_folder):
            continue

        embeddings = []

        for file in os.listdir(student_folder):

            if file.endswith((".jpg",".jpeg",".png")):

                img_path = os.path.join(student_folder,file)

                img = cv2.imread(img_path)

                try:

                    rep = DeepFace.represent(
                        img_path=img,
                        model_name="Facenet",
                        enforce_detection=False
                    )

                    embedding = rep[0]["embedding"]
                    embeddings.append(embedding)

                except:
                    continue

        if len(embeddings) > 0:

            # average embedding of student
            avg_embedding = np.mean(embeddings, axis=0)
            database[student_name] = avg_embedding

    print(f"✅ Loaded {len(database)} students into database")

    return database


# ==============================
# Cosine similarity
# ==============================
def cosine_similarity(a,b):

    a = np.array(a)
    b = np.array(b)

    return np.dot(a,b) / (np.linalg.norm(a)*np.linalg.norm(b))


# ==============================
# Recognize face
# ==============================
def recognize_face(face, database):

    try:

        rep = DeepFace.represent(
            img_path=face,
            model_name="Facenet",
            enforce_detection=False
        )

        embedding = rep[0]["embedding"]

    except:
        return "Unknown"

    best_match = None
    best_score = -1

    for name,db_embedding in database.items():

        score = cosine_similarity(embedding, db_embedding)

        if score > best_score:

            best_score = score
            best_match = name

    if best_score > 0.55:
        return best_match
    else:
        return "Unknown"