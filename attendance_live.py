
import cv2
import time
import threading

import argparse
import os

from insightface.app import FaceAnalysis

from recognition import FaceRecognizer
from attendance_manager import AttendanceManager


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

STUDENT_NAMES = {
    "230108060": "Vedant",
    "230108061": "Surbhi"
}

CAMERA_INDEX = 0
DETECTION_SIZE = (320, 320)
OVERLAY_DURATION = 1.0


# --------------------------------------------------
# SHARED VARIABLES
# --------------------------------------------------

latest_frame = None
frame_id = 0

latest_detections = []
latest_result_time = 0

inference_time = 0

# Stores attendance status for each student
attendance_status = {}

# Students whose attendance has already been recorded
attendance_attempted = set()

lock = threading.Lock()
stop_event = threading.Event()


# --------------------------------------------------
# INITIALIZE FACE RECOGNITION
# --------------------------------------------------

def initialize_recognition():

    print("\nLoading InsightFace model...")

    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"]
    )

    app.prepare(
        ctx_id=-1,
        det_size=DETECTION_SIZE
    )

    print("InsightFace loaded.")

    print("\nLoading SVM classifier...")

    recognizer = FaceRecognizer()

    print("SVM classifier loaded.")

    return app, recognizer


# --------------------------------------------------
# SELECT COURSE AND START SESSION
# --------------------------------------------------

def select_course_and_start(db):

    courses = db.get_courses()

    print("\n========== AVAILABLE COURSES ==========\n")

    for course in courses:

        print(
            f"{course['course_id']} - "
            f"{course['course_name']}"
        )

    try:

        course_id = int(
            input("\nEnter course ID: ")
        )

    except ValueError:

        print("Invalid course ID.")
        return None

    selected_course = next(
        (
            course for course in courses
            if course["course_id"] == course_id
        ),
        None
    )

    if selected_course is None:

        print("Course does not exist.")
        return None

    # Start attendance session
    success, message, session_id = db.start_attendance(
        course_id
    )

    print("\n" + message)

    if not success:
        return None

    print("Session ID:", session_id)
    print("Selected Course:", selected_course["course_name"])

    return session_id, selected_course["course_name"]


# --------------------------------------------------
# CAMERA WORKER
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
# RECOGNITION + ATTENDANCE WORKER
# --------------------------------------------------

def recognition_worker(app, recognizer, db, session_id):

    global latest_detections
    global latest_result_time
    global inference_time

    last_processed_frame_id = -1

    while not stop_event.is_set():

        # Get latest camera frame
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

        # Avoid processing same frame repeatedly
        if current_frame_id == last_processed_frame_id:

            time.sleep(0.005)
            continue

        last_processed_frame_id = current_frame_id

        start_time = time.time()

        # Detect faces
        faces = app.get(frame)

        detections = []

        for face in faces:

            # Extract ArcFace embedding
            embedding = face.normed_embedding

            # Predict student ID
            student_id, similarity = recognizer.recognize(
                embedding
            )

            student_id = str(student_id)

            # --------------------------------------
            # UNKNOWN PERSON
            # --------------------------------------

            if student_id == "Unknown":

                name = "Unknown"
                status = "UNKNOWN"
                color = (0, 0, 255)

            # --------------------------------------
            # REGISTERED STUDENT
            # --------------------------------------

            else:

                name = STUDENT_NAMES.get(
                    student_id,
                    student_id
                )

                # Record attendance only once per student
                with lock:

                    already_attempted = (
                        student_id in attendance_attempted
                    )

                if not already_attempted:

                    success, message = db.mark_attendance(
                        student_id,
                        session_id
                    )

                    print(
                        f"\n{student_id}: {message}"
                    )

                    if success or "already marked" in message.lower():

                        with lock:

                            attendance_attempted.add(student_id)

                            attendance_status[student_id] = "PRESENT"

                # Read current attendance status
                with lock:

                    status = attendance_status.get(
                        student_id,
                        "PENDING"
                    )

                if status == "PRESENT":

                    color = (0, 255, 0)

                else:

                    color = (0, 165, 255)

            # --------------------------------------
            # STORE DETECTION
            # --------------------------------------

            x1, y1, x2, y2 = map(
                int,
                face.bbox
            )

            detections.append({

                "bbox": (x1, y1, x2, y2),

                "name": name,

                "student_id": student_id,

                "similarity": float(similarity),

                "status": status,

                "color": color

            })

        elapsed = time.time() - start_time

        with lock:

            latest_detections = detections

            latest_result_time = time.time()

            inference_time = elapsed


# --------------------------------------------------
# DISPLAY WORKER
# --------------------------------------------------

def display_worker(stop_file=None):

    global inference_time

    previous_time = time.time()
    fps = 0

    while not stop_event.is_set():

        with lock:
            
            if stop_file and os.path.exists(stop_file):
                stop_event.set()
                break

            if latest_frame is None:
                frame = None

            else:
                frame = latest_frame.copy()

            detections = latest_detections.copy()

            result_time = latest_result_time

            current_inference_time = inference_time

        if frame is None:

            time.sleep(0.01)
            continue

        display_frame = frame.copy()

        # --------------------------------------
        # DRAW FACE BOXES
        # --------------------------------------

        if time.time() - result_time < OVERLAY_DURATION:

            for detection in detections:

                x1, y1, x2, y2 = detection["bbox"]

                name = detection["name"]

                student_id = detection["student_id"]

                similarity = detection["similarity"]

                status = detection["status"]

                color = detection["color"]

                # Draw bounding box
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

                # Draw label background
                (text_width, text_height), baseline = cv2.getTextSize(
                    label,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    2
                )

                label_y = max(
                    y1 - 10,
                    text_height + 10
                )

                cv2.rectangle(
                    display_frame,
                    (
                        x1,
                        label_y - text_height - 10
                    ),
                    (
                        x1 + text_width + 10,
                        label_y + baseline - 5
                    ),
                    color,
                    -1
                )

                cv2.putText(
                    display_frame,
                    label,
                    (x1 + 5, label_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2
                )

                # Display attendance status
                if student_id != "Unknown":

                    cv2.putText(
                        display_frame,
                        status,
                        (x1, y2 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        color,
                        2
                    )

        # --------------------------------------
        # DISPLAY ATTENDANCE COUNTS
        # --------------------------------------

        with lock:

            total_marked = sum(
                1 for status in attendance_status.values()
                if status == "PRESENT"
            )

        cv2.putText(
            display_frame,
            f"Students Present: {total_marked}",
            (10, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )

        # --------------------------------------
        # CALCULATE FPS
        # --------------------------------------

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

        # --------------------------------------
        # DISPLAY PERFORMANCE
        # --------------------------------------

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

        cv2.imshow(
            "Face Recognition - Attendance System",
            display_frame
        )

        # Press Q to stop attendance
        if cv2.waitKey(1) & 0xFF == ord("q"):

            stop_event.set()
            break


# --------------------------------------------------
# MAIN FUNCTION
# --------------------------------------------------

def main(course_id=None, stop_file=None):

    db = AttendanceManager()

    cap = None
    session_id = None

    try:

        # Initialize recognition models
        app, recognizer = initialize_recognition()

        # Select course and start session
        if course_id is None:
            result = select_course_and_start(db)

        else:

            selected_course = next(
                (
                    course for course in db.get_courses()
                    if course["course_id"] == course_id
                ),
                None
            )

            if selected_course is None:
                print("Invalid course ID.")
                return

            success, message, session_id = db.start_attendance(
                course_id
            )

            print(message)

            if not success:
                return

            result = (
                session_id,
                selected_course["course_name"]
            )

        if result is None:
            return

        session_id, course_name = result

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

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        print("\n===================================")
        print("ATTENDANCE STARTED")
        print("Course:", course_name)
        print("Session ID:", session_id)
        print("Press Q to end attendance")
        print("===================================\n")

        # Start camera thread
        camera_thread = threading.Thread(
            target=camera_worker,
            args=(cap,),
            daemon=True
        )

        # Start recognition and attendance thread
        recognition_thread = threading.Thread(
            target=recognition_worker,
            args=(app, recognizer, db, session_id),
            daemon=True
        )

        camera_thread.start()
        recognition_thread.start()

        # Display loop
        display_worker(stop_file)

        # Stop worker threads
        stop_event.set()

        camera_thread.join(timeout=1)
        recognition_thread.join(timeout=2)

        # Display final attendance
        print("\n========== SESSION ATTENDANCE ==========\n")

        records = db.get_session_attendance(session_id)

        for record in records:

            print(
                record["student_id"],
                "-",
                record["name"],
                "-",
                record["marked_at"]
            )

        print(
            "\nTotal students present:",
            db.get_attendance_count(session_id)
        )

    except KeyboardInterrupt:

        print("\nAttendance interrupted.")

    finally:

        stop_event.set()

        if cap is not None:
            cap.release()

        cv2.destroyAllWindows()

        # End the attendance session
        if session_id is not None:

            success, message = db.end_attendance(session_id)

            print("\n" + message)

        db.close()


# --------------------------------------------------
# ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--course-id",
        type=int,
        required=True
    )

    parser.add_argument(
        "--stop-file",
        type=str,
        required=True
    )

    args = parser.parse_args()

    main(
        course_id=args.course_id,
        stop_file=args.stop_file
    )
