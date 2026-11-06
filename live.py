
import cv2
import time
import threading
import numpy as np

from insightface.app import FaceAnalysis
from recognition import FaceRecognizer


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

STUDENT_NAMES = {
    "230108060": "Vedant",
    "230108061": "Surbhi"
}

CAMERA_INDEX = 0
DETECTION_SIZE = (320, 320)

# How long to retain the latest detection overlay
OVERLAY_DURATION = 1.0


# --------------------------------------------------
# SHARED VARIABLES
# --------------------------------------------------

latest_frame = None
frame_id = 0

latest_detections = []
latest_result_time = 0

inference_time = 0

lock = threading.Lock()
stop_event = threading.Event()


# --------------------------------------------------
# INITIALIZE INSIGHTFACE
# --------------------------------------------------

print("Loading InsightFace model...")

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(
    ctx_id=-1,
    det_size=DETECTION_SIZE
)

print("InsightFace loaded successfully.")


# --------------------------------------------------
# INITIALIZE FACE RECOGNIZER
# --------------------------------------------------

print("Loading SVM classifier...")

recognizer = FaceRecognizer()

print("Face recognizer loaded successfully.")


# --------------------------------------------------
# CAMERA WORKER
# Continuously captures the latest webcam frame.
# --------------------------------------------------

def camera_worker(cap):

    global latest_frame, frame_id

    while not stop_event.is_set():

        ret, frame = cap.read()

        if not ret:
            time.sleep(0.005)
            continue

        with lock:
            latest_frame = frame
            frame_id += 1


# --------------------------------------------------
# RECOGNITION WORKER
# Detects faces and identifies students.
# --------------------------------------------------

def recognition_worker():

    global latest_detections
    global latest_result_time
    global inference_time

    last_processed_frame_id = -1

    while not stop_event.is_set():

        with lock:

            if latest_frame is None:
                frame = None
                current_frame_id = -1
            else:
                frame = latest_frame.copy()
                current_frame_id = frame_id

        if frame is None:
            time.sleep(0.01)
            continue

        # Avoid processing the same frame repeatedly
        if current_frame_id == last_processed_frame_id:
            time.sleep(0.005)
            continue

        last_processed_frame_id = current_frame_id

        start_time = time.time()

        # Detect faces
        faces = app.get(frame)

        detections = []

        for face in faces:

            # Extract 512-dimensional ArcFace embedding
            embedding = face.normed_embedding

            # Recognize student using trained SVM
            student_id, similarity = recognizer.recognize(
                embedding
            )

            student_id = str(student_id)

            # Determine display name and box color
            if student_id == "Unknown":

                name = "Unknown"
                color = (0, 0, 255)

            else:

                name = STUDENT_NAMES.get(
                    student_id,
                    "Unknown"
                )

                color = (
                    (0, 255, 0)
                    if name != "Unknown"
                    else (0, 0, 255)
                )

            # Bounding box coordinates
            x1, y1, x2, y2 = map(
                int,
                face.bbox
            )

            detections.append({

                "bbox": (x1, y1, x2, y2),

                "name": name,

                "student_id": student_id,

                "similarity": similarity,

                "color": color

            })

        elapsed = time.time() - start_time

        with lock:

            latest_detections = detections

            latest_result_time = time.time()

            inference_time = elapsed


# --------------------------------------------------
# DISPLAY WORKER
# Displays smooth video and recognition overlays.
# --------------------------------------------------

def display_worker():

    global inference_time

    previous_time = time.time()

    fps = 0

    while not stop_event.is_set():

        with lock:

            if latest_frame is None:
                frame = None
            else:
                frame = latest_frame.copy()

            detections_to_draw = latest_detections.copy()

            result_time = latest_result_time

            current_inference_time = inference_time

        if frame is None:
            time.sleep(0.01)
            continue

        display_frame = frame.copy()

        # --------------------------------------------------
        # DRAW FACE DETECTION BOXES
        # --------------------------------------------------

        if time.time() - result_time < OVERLAY_DURATION:

            for detection in detections_to_draw:

                x1, y1, x2, y2 = detection["bbox"]

                name = detection["name"]

                student_id = detection["student_id"]

                similarity = detection["similarity"]

                color = detection["color"]

                # Draw rectangle around face
                cv2.rectangle(
                    display_frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2
                )

                # Prepare label
                label = f"{name} ({similarity:.2f})"

                if student_id != "Unknown":
                    label += f" | {student_id}"

                # Calculate label dimensions
                (text_width, text_height), baseline = cv2.getTextSize(
                    label,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    2
                )

                # Prevent label background from going outside frame
                label_y = max(y1 - 10, text_height + 10)

                # Draw label background
                cv2.rectangle(
                    display_frame,
                    (x1, label_y - text_height - 10),
                    (x1 + text_width + 10, label_y + baseline - 5),
                    color,
                    -1
                )

                # Draw label text
                cv2.putText(
                    display_frame,
                    label,
                    (x1 + 5, label_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2
                )

        # --------------------------------------------------
        # CALCULATE DISPLAY FPS
        # --------------------------------------------------

        current_time = time.time()

        delta = current_time - previous_time

        if delta > 0:
            instant_fps = 1.0 / delta

            fps = (
                0.9 * fps + 0.1 * instant_fps
                if fps > 0
                else instant_fps
            )

        previous_time = current_time

        # --------------------------------------------------
        # DISPLAY PERFORMANCE INFORMATION
        # --------------------------------------------------

        cv2.putText(
            display_frame,
            f"Display FPS: {fps:.1f}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2
        )

        cv2.putText(
            display_frame,
            f"Inference: {current_inference_time * 1000:.0f} ms",
            (10, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2
        )

        # --------------------------------------------------
        # SHOW VIDEO
        # --------------------------------------------------

        cv2.imshow(
            "ArcFace Student Recognition",
            display_frame
        )

        # Press Q to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):

            stop_event.set()
            break


# --------------------------------------------------
# MAIN FUNCTION
# --------------------------------------------------

def main():

    print("\nStarting webcam...")

    cap = cv2.VideoCapture(
        CAMERA_INDEX,
        cv2.CAP_DSHOW
    )

    if not cap.isOpened():

        print("DirectShow failed. Trying default backend...")

        cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():

        print("ERROR: Could not open webcam.")
        return

    # Webcam configuration
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print("Webcam started.")

    print("\nControls:")
    print("Q - Quit")

    # Start camera thread
    camera_thread = threading.Thread(
        target=camera_worker,
        args=(cap,),
        daemon=True
    )

    # Start recognition thread
    recognition_thread = threading.Thread(
        target=recognition_worker,
        daemon=True
    )

    camera_thread.start()
    recognition_thread.start()

    try:

        # Display loop runs on main thread
        display_worker()

    except KeyboardInterrupt:

        print("\nStopping application...")

    finally:

        stop_event.set()

        cap.release()

        cv2.destroyAllWindows()

        camera_thread.join(timeout=1)
        recognition_thread.join(timeout=2)

        print("Application stopped.")


# --------------------------------------------------
# ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":

    main()
