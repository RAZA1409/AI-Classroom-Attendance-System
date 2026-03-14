import os
import cv2
import numpy as np
from deepface import DeepFace


# ==============================
# Load student embeddings
# ==============================
def load_student_database(base_path="students"):

    database = {}

    if not os.path.exists(base_path):
        print("❌ Students folder not found")
        return database

    for student_name in os.listdir(base_path):

        student_folder = os.path.join(base_path, student_name)

        if not os.path.isdir(student_folder):
            continue

        embeddings = []

        for file in os.listdir(student_folder):

            if file.lower().endswith((".jpg", ".jpeg", ".png")):

                img_path = os.path.join(student_folder, file)

                img = cv2.imread(img_path)

                if img is None:
                    continue

                # Resize image for stable embeddings
                img = cv2.resize(img, (160, 160))

                try:

                    rep = DeepFace.represent(
                        img_path=img,
                        model_name="Facenet512",
                        enforce_detection=False
                    )

                    embedding = rep[0]["embedding"]
                    embeddings.append(embedding)

                except Exception as e:
                    print("Embedding error:", e)
                    continue

        if len(embeddings) > 0:

            # Average embedding for student
            avg_embedding = np.mean(embeddings, axis=0)
            database[student_name] = avg_embedding

    print(f"✅ Loaded {len(database)} students into database")

    return database


# ==============================
# Cosine similarity
# ==============================
def cosine_similarity(a, b):

    a = np.array(a)
    b = np.array(b)

    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# ==============================
# Recognize face
# ==============================
def recognize_face(face, database):

    if face is None or face.size == 0:
        return "Unknown"

    try:

        # Resize face to match dataset embeddings
        face = cv2.resize(face, (160, 160))

        rep = DeepFace.represent(
            img_path=face,
            model_name="Facenet512",
            enforce_detection=False
        )

        embedding = rep[0]["embedding"]

    except Exception as e:
        print("Face embedding error:", e)
        return "Unknown"

    best_match = None
    best_score = -1

    for name, db_embedding in database.items():

        score = cosine_similarity(embedding, db_embedding)

        if score > best_score:

            best_score = score
            best_match = name

    # Debug print (VERY useful)
    print("Best match:", best_match, "| Score:", round(best_score, 3))

    # Threshold
    if best_score > 0.60:
        return best_match
    else:
        return "Unknown"