import os
import pickle
import numpy as np
import cv2
from deepface import DeepFace


def load_database(path="face_embeddings.pkl"):
    """
    Safely load pre-computed face embeddings from a pickle file.
    Validates file existence and dictionary structure.
    """
    if not os.path.exists(path):
        print(f"⚠️ Face database file '{path}' not found. Run build_face_database.py to create it.")
        return {}

    try:
        with open(path, "rb") as f:
            database = pickle.load(f)

        if not isinstance(database, dict):
            print(f"⚠️ Face database '{path}' does not contain a valid dictionary.")
            return {}

        print("Loaded students:", len(database))
        return database
    except Exception as e:
        print(f"❌ Failed to load face database from '{path}': {e}")
        return {}


def cosine_similarity(a, b):
    """
    Compute cosine similarity with zero-division safeguard.
    """
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))


def recognize_face(face, database, debug=False):
    """
    Recognize a face against the loaded embedding database.
    """
    if not database:
        return "Unknown"

    try:
        face = cv2.resize(face, (160, 160))

        rep = DeepFace.represent(
            img_path=face,
            model_name="Facenet512",
            enforce_detection=False
        )

        embedding = rep[0]["embedding"]

    except Exception:
        return "Unknown"

    best_match = None
    best_score = -1

    for name, embeddings in database.items():
        for db_emb in embeddings:
            score = cosine_similarity(embedding, db_emb)

            if score > best_score:
                best_score = score
                best_match = name

    if debug and best_match:
        print("Best match:", best_match, "Score:", round(best_score, 3))

    if best_score > 0.6:
        return best_match
    else:
        return "Unknown"
