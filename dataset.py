
import cv2
import os
import time
import threading

from insightface.app import FaceAnalysis


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

DATASET_DIR = "dataset"
MODEL_NAME = "buffalo_l"

CAMERA_INDEX = 0

TARGET_IMAGES = 100

DETECTION_INTERVAL = 3
DETECTION_SIZE = (320, 320)

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

CAPTURE_INTERVAL = 0.5

# Reject detection results older than this
MAX_DETECTION_AGE = 1.0


# --------------------------------------------------
# SHARED VARIABLES
# --------------------------------------------------

stop_event = threading.Event()
frame_lock = threading.Lock()

latest_frame = None
latest_frame_id = 0

detected_faces = []
detection_frame = None
detection_timestamp = 0

camera_error = None


# --------------------------------------------------
# INITIALIZE INSIGHTFACE
# --------------------------------------------------

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


# --------------------------------------------------
# THREAD 1: CAMERA CAPTURE
# --------------------------------------------------

def camera_worker(cap):

    global latest_frame
    global latest_frame_id
    global camera_error

    while not stop_event.is_set():

        ret, frame = cap.read()

        if not ret or frame is None:

            camera_error = "Unable to read webcam frame."
            stop_event.set()
            break

        with frame_lock:

            latest_frame = frame
            latest_frame_id += 1


# --------------------------------------------------
# THREAD 2: FACE DETECTION
# --------------------------------------------------

def detection_worker(app):

    global detected_faces
    global detection_frame
    global detection_timestamp

    last_processed_id = -1

    while not stop_event.is_set():

        with frame_lock:

            if latest_frame is None:
                frame = None
                frame_id = -1

            else:
                frame = latest_frame.copy()
                frame_id = latest_frame_id

        if frame is None:

            time.sleep(0.005)
            continue

        # Avoid processing the same frame repeatedly
        if frame_id == last_processed_id:

            time.sleep(0.005)
            continue

        last_processed_id = frame_id

        # Run detection only every N frames
        if frame_id % DETECTION_INTERVAL != 0:
            continue

        try:

            faces = app.get(frame)

            face_data = []

            for face in faces:

                x1, y1, x2, y2 = map(
                    int,
                    face.bbox
                )

                score = float(face.det_score)

                face_data.append({
                    "bbox": (x1, y1, x2, y2),
                    "score": score
                })

            with frame_lock:

                detected_faces = face_data

                # Keep the exact frame used for detection.
                # This ensures the saved image matches its
                # corresponding face detection result.
                detection_frame = frame.copy()

                detection_timestamp = time.monotonic()

        except Exception as e:

            print("Face detection error:", e)


# --------------------------------------------------
# DATASET GENERATION
# --------------------------------------------------

def generate_dataset():

    global detected_faces
    global detection_frame
    global detection_timestamp

    app = initialize_insightface()

    student_id = input(
        "Enter student roll number: "
    ).strip()

    if not student_id:

        print("Invalid student ID.")
        return

    student_dir = os.path.join(
        DATASET_DIR,
        student_id
    )

    os.makedirs(student_dir, exist_ok=True)

    existing_images = [
        f for f in os.listdir(student_dir)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]

    img_id = len(existing_images)

    if img_id >= TARGET_IMAGES:

        print("Target image count already reached.")
        return

    # --------------------------------------------------
    # OPEN CAMERA
    # --------------------------------------------------

    cap = cv2.VideoCapture(
        CAMERA_INDEX,
        cv2.CAP_DSHOW
    )

    if not cap.isOpened():

        print("DirectShow failed. Trying default backend.")

        cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():

        print("Error: Cannot open webcam.")
        return

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        CAMERA_WIDTH
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        CAMERA_HEIGHT
    )

    cap.set(
        cv2.CAP_PROP_BUFFERSIZE,
        1
    )

    # Reset shared variables
    stop_event.clear()

    with frame_lock:

        global latest_frame
        global latest_frame_id
        global camera_error

        latest_frame = None
        latest_frame_id = 0

        detected_faces = []
        detection_frame = None
        detection_timestamp = 0

        camera_error = None

    # --------------------------------------------------
    # START THREADS
    # --------------------------------------------------

    camera_thread = threading.Thread(
        target=camera_worker,
        args=(cap,),
        daemon=True
    )

    detection_thread = threading.Thread(
        target=detection_worker,
        args=(app,),
        daemon=True
    )

    camera_thread.start()
    detection_thread.start()

    print("\nDataset collection started.")
    print("C = Capture image")
    print("Q = Quit")

    last_capture_time = 0

    # --------------------------------------------------
    # MAIN DISPLAY LOOP
    # --------------------------------------------------

    try:

        while img_id < TARGET_IMAGES:

            if camera_error:

                print(camera_error)
                break

            with frame_lock:

                if latest_frame is None:
                    frame = None
                else:
                    frame = latest_frame.copy()

                faces = list(detected_faces)

                saved_frame = (
                    detection_frame.copy()
                    if detection_frame is not None
                    else None
                )

                detection_time = detection_timestamp

            if frame is None:

                time.sleep(0.005)
                continue

            display_frame = frame.copy()

            # --------------------------------------------------
            # DRAW FACE BOUNDARIES
            # --------------------------------------------------

            for face in faces:

                x1, y1, x2, y2 = face["bbox"]

                score = face["score"]

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

            # --------------------------------------------------
            # DISPLAY INFORMATION
            # --------------------------------------------------

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
                f"Faces detected: {len(faces)}",
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

            # --------------------------------------------------
            # QUIT
            # --------------------------------------------------

            if key == ord("q"):

                break

            # --------------------------------------------------
            # CAPTURE IMAGE
            # --------------------------------------------------

            if key == ord("c"):

                current_time = time.monotonic()

                if current_time - last_capture_time < CAPTURE_INTERVAL:

                    continue

                # Check whether detection result is fresh
                if (
                    saved_frame is None
                    or detection_time == 0
                    or current_time - detection_time > MAX_DETECTION_AGE
                ):

                    print("Please wait for fresh face detection.")
                    continue

                # Exactly one face must be detected
                if len(faces) != 1:

                    print(
                        "Capture rejected: "
                        "Exactly one face required."
                    )

                    continue

                face = faces[0]

                image_path = os.path.join(
                    student_dir,
                    f"{img_id + 1:03d}.jpg"
                )

                success = cv2.imwrite(
                    image_path,
                    saved_frame
                )

                if success:

                    img_id += 1

                    last_capture_time = current_time

                    print(
                        f"Saved image {img_id}/{TARGET_IMAGES} "
                        f"| Detection score: {face['score']:.3f}"
                    )

        print(
            f"\nDataset collection finished: {img_id} images."
        )

    finally:

        # Stop both threads
        stop_event.set()

        camera_thread.join(timeout=2)
        detection_thread.join(timeout=2)

        cap.release()

        cv2.destroyAllWindows()


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":

    generate_dataset()
