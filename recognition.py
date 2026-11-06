
import os
import joblib
import numpy as np


# ---------------- CONFIGURATION ----------------

MODEL_DIR = "models"

CLASSIFIER_FILE = os.path.join(
    MODEL_DIR, "svm_classifier.pkl"
)

EMBEDDING_FILE = os.path.join(
    MODEL_DIR, "embeddings.npz"
)

# Experimental starting threshold.
# Must be validated using separate webcam sessions
# and images of people who are not registered.
MIN_SIMILARITY = 0.45

# ------------------------------------------------


class FaceRecognizer:

    def __init__(self):

        print("Loading SVM classifier...")

        self.classifier = joblib.load(CLASSIFIER_FILE)

        print("Loading student embeddings...")

        data = np.load(EMBEDDING_FILE)

        X = data["embeddings"]
        y = data["labels"]

        self.student_centroids = {}

        # Calculate one normalized centroid per student
        for student_id in np.unique(y):

            student_embeddings = X[y == student_id]

            centroid = np.mean(
                student_embeddings,
                axis=0
            )

            centroid_norm = np.linalg.norm(centroid)

            if centroid_norm == 0:
                continue

            centroid = centroid / centroid_norm

            self.student_centroids[student_id] = centroid

        print("SVM loaded successfully.")

        print(
            "Registered students:",
            list(self.student_centroids.keys())
        )

    def recognize(self, embedding):

        embedding = np.asarray(
            embedding,
            dtype=np.float32
        ).reshape(1, -1)

        # Predict identity using SVM
        predicted_id = self.classifier.predict(embedding)[0]

        # Compare against registered student centroids
        similarities = {}

        for student_id, centroid in self.student_centroids.items():

            similarity = float(
                np.dot(embedding[0], centroid)
                / np.linalg.norm(embedding[0])
            )

            similarities[student_id] = similarity

        # Similarity of the SVM-predicted student
        similarity = similarities.get(predicted_id, 0.0)

        # Reject low-similarity predictions
        if similarity < MIN_SIMILARITY:

            return "Unknown", similarity

        return predicted_id, similarity
