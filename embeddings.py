
import cv2
import os
import numpy as np
from insightface.app import FaceAnalysis


# ---------------- CONFIGURATION ----------------

DATASET_DIR = "dataset"
MODEL_DIR = "models"

MODEL_NAME = "buffalo_l"
DETECTION_SIZE = (320, 320)

OUTPUT_FILE = os.path.join(MODEL_DIR, "embeddings.npz")

# ------------------------------------------------


def initialize_insightface():

    app = FaceAnalysis(
        name=MODEL_NAME,
        providers=["CPUExecutionProvider"]
    )

    app.prepare(
        ctx_id=-1,
        det_size=DETECTION_SIZE
    )

    return app


def extract_embeddings():

    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Initializing InsightFace...")

    app = initialize_insightface()

    embeddings = []
    labels = []

    total_images = 0
    processed_images = 0
    skipped_images = 0

    # Get all student directories
    student_folders = [
        folder for folder in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, folder))
    ]

    if not student_folders:
        print("No student datasets found.")
        return

    print(f"\nFound {len(student_folders)} student folders.")

    for student_id in student_folders:

        student_path = os.path.join(DATASET_DIR, student_id)

        print(f"\nProcessing student: {student_id}")

        image_files = [
            file for file in os.listdir(student_path)
            if file.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        for image_file in image_files:

            total_images += 1

            image_path = os.path.join(student_path, image_file)

            # Read image
            img = cv2.imread(image_path)

            if img is None:
                print(f"Skipping unreadable image: {image_file}")
                skipped_images += 1
                continue

            # Detect faces
            faces = app.get(img)

            # Require exactly one face
            if len(faces) != 1:
                print(
                    f"Skipping {image_file}: "
                    f"Expected 1 face, found {len(faces)}"
                )
                skipped_images += 1
                continue

            face = faces[0]

            # Extract normalized ArcFace embedding
            embedding = face.normed_embedding

            if embedding is None:
                print(f"Embedding unavailable: {image_file}")
                skipped_images += 1
                continue

            embedding = np.asarray(embedding, dtype=np.float32)

            if embedding.shape != (512,) or not np.all(np.isfinite(embedding)):
                print(f"Invalid embedding: {image_file}")
                skipped_images += 1
                continue

            embeddings.append(embedding)
            labels.append(student_id)

            processed_images += 1

            print(
                f"Processed: {student_id}/{image_file}"
            )

    if not embeddings:
        print("\nNo valid embeddings generated.")
        return

    # Convert to NumPy arrays
    X = np.asarray(embeddings, dtype=np.float32)
    y = np.asarray(labels, dtype=str)

    # Save embeddings and labels
    np.savez_compressed(
        OUTPUT_FILE,
        embeddings=X,
        labels=y
    )

    print("\n========== EMBEDDING SUMMARY ==========")

    print("Total images:", total_images)
    print("Successfully processed:", processed_images)
    print("Skipped images:", skipped_images)

    print("Embedding matrix shape:", X.shape)
    print("Labels shape:", y.shape)

    print("Embedding dimensions:", X.shape[1])

    print("Saved to:", OUTPUT_FILE)

    print("\nEmbedding extraction completed successfully!")


if __name__ == "__main__":
    extract_embeddings()
