
import cv2
import os
import time
from insightface.app import FaceAnalysis


# ---------------- CONFIGURATION ----------------

DATASET_DIR = "dataset"
MODEL_NAME = "buffalo_l"

CAMERA_INDEX = 0

TARGET_IMAGES = 100

# Run detection once every N frames
DETECTION_INTERVAL = 3

# Lower detection resolution
DETECTION_SIZE = (320, 320)

# Webcam resolution
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# Minimum time between captures
CAPTURE_INTERVAL = 0.5

# ------------------------------------------------


def initialize_insightface():

    print("Initializing InsightFace...")

    app = FaceAnalysis(
        name=MODEL_NAME,
        providers=["CPUExecutionProvider"]
    )

    app.prepare(
        ctx_id=-1,
        det_size=DETECTION_SIZE
    )

    print("InsightFace initialized successfully.")

    return app


def generate_dataset():

    app = initialize_insightface()

    student_id = input("Enter student roll number: ").strip()

    if not student_id:
        print("Invalid student ID.")
        return

    student_dir = os.path.join(DATASET_DIR, student_id)

    os.makedirs(student_dir, exist_ok=True)

    existing_images = [
        f for f in os.listdir(student_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    img_id = len(existing_images)

    if img_id >= TARGET_IMAGES:
        print("Target image count already reached.")
        return

    # Use DirectShow on Windows
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("DirectShow failed. Trying default camera backend.")
        cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("Error: Cannot open webcam.")
        return

    # Configure webcam
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    # Request a small camera buffer
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print("\nDataset collection started.")
    print("C = Capture image")
    print("Q = Quit")

    frame_count = 0

    detected_faces = []
    last_capture_time = 0

    try:

        while img_id < TARGET_IMAGES:

            ret, frame = cap.read()

            if not ret or frame is None:
                print("Camera frame unavailable.")
                break

            frame_count += 1

            # Run detection periodically
            if frame_count % DETECTION_INTERVAL == 0:

                detected_faces = app.get(frame)

            display_frame = frame.copy()

            # Draw previously detected faces
            for face in detected_faces:

                x1, y1, x2, y2 = map(int, face.bbox)

                score = float(face.det_score)

                cv2.rectangle(
                    display_frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    display_frame,
                    f"Face: {score:.2f}",
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

            # Display information
            cv2.putText(
                display_frame,
                f"Student: {student_id}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                display_frame,
                f"Images: {img_id}/{TARGET_IMAGES}",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.putText(
                display_frame,
                f"Faces detected: {len(detected_faces)}",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )

            cv2.imshow(
                "InsightFace Dataset Collection",
                display_frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            # Capture only on C
            if key == ord("c"):

                current_time = time.time()

                if current_time - last_capture_time < CAPTURE_INTERVAL:
                    continue

                if len(detected_faces) != 1:
                    print("Capture rejected: Exactly one face required.")
                    continue

                face = detected_faces[0]

                # No arbitrary 0.80 confidence rejection.
                # Display and inspect the actual detection score.

                image_path = os.path.join(
                    student_dir,
                    f"{img_id + 1:03d}.jpg"
                )

                success = cv2.imwrite(image_path, frame)

                if success:

                    img_id += 1
                    last_capture_time = current_time

                    print(
                        f"Saved image {img_id}/{TARGET_IMAGES} "
                        f"| Detection score: {face.det_score:.3f}"
                    )

        print(f"\nDataset collection finished: {img_id} images.")

    finally:

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    generate_dataset()
