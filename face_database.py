import pickle
import numpy as np
import cv2
from deepface import DeepFace


def load_database(path="face_embeddings.pkl"):

    with open(path, "rb") as f:
        database = pickle.load(f)

    print("Loaded students:", len(database))
    return database


def cosine_similarity(a, b):

    a = np.array(a)
    b = np.array(b)

    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def recognize_face(face, database):

    try:

        face = cv2.resize(face, (160,160))

        rep = DeepFace.represent(
            img_path=face,
            model_name="Facenet512",
            enforce_detection=False
        )

        embedding = rep[0]["embedding"]

    except:
        return "Unknown"

    best_match = None
    best_score = -1

    for name, embeddings in database.items():

        for db_emb in embeddings:

            score = cosine_similarity(embedding, db_emb)

            if score > best_score:
                best_score = score
                best_match = name

    print("Best:", best_match, "Score:", round(best_score,3))

    if best_score > 0.6:
        return best_match
    else:
        return "Unknown"