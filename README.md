# Face Recognition Attendance System

A Python attendance application that uses InsightFace/ArcFace face embeddings, an
SVM classifier, OpenCV, Streamlit, and SQLite.

## Features

- Register students and automatically enrol them in the four default courses.
- Add, enable, disable, and delete course schedules.
- Capture student face images using a webcam.
- Run course attendance sessions with live face recognition in an OpenCV window.
- View attendance summaries and daily and detailed reports.
- Filter students below a selected attendance percentage and download CSV reports.

## Project files

```text
.
├── .streamlit/
│   └── config.toml
├── attendance/             # SQLite database, reports, and session controls
├── dataset/                # Captured face images, organized by student ID
├── models/                 # Generated embeddings and trained model
├── app.py                  # Streamlit dashboard
├── attendance_live.py      # Live attendance process
├── attendance_manager.py   # SQLite database and attendance operations
├── dataset.py              # Webcam face-image capture
├── embeddings.py           # Face embedding extraction
├── train.py                # SVM training and evaluation
├── recognition.py          # Face recognition helper
├── live.py
├── evaluate.py
├── preprocessing.py
└── requirements.txt
```

The `attendance`, `dataset`, and `models` directories contain local data or
generated files. They are not included in Git.

## Setup

Run these commands from the project directory in PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt streamlit pandas
```

`app.py` uses Streamlit and pandas; these packages are not currently listed in
`requirements.txt`.

The first run may download InsightFace's `buffalo_l` model. A working webcam is
needed for image capture and live attendance.

## Start the dashboard

```powershell
streamlit run app.py
```

Open the local URL printed by Streamlit. The dashboard includes **Dashboard**,
**Students**, **Schedules**, **Live Attendance**, **Reports**, and
**Attendance Shortage** pages. Live attendance opens in a separate OpenCV
window. End a session using the dashboard's **End Attendance** control.

## Register a student and update the recognition model

1. Register the student's ID and name on the dashboard's **Students** page.
2. In a second terminal, activate the virtual environment and capture images:

   ```powershell
   .\venv\Scripts\Activate.ps1
   python dataset.py
   ```

   Enter the student's ID when prompted. Press **C** to capture an image and
   **Q** to stop. Images are saved under `dataset/<student_id>/`.

3. Generate embeddings and train the classifier:

   ```powershell
   python embeddings.py
   python train.py
   ```

   Training requires images for at least two student IDs. The generated
   embeddings, classifier, and confusion-matrix image are saved under `models/`.
4. Restart any running live recognition process so it loads the updated model.

## Attendance data and reports

The SQLite database is created at `attendance/attendance.db`. Course CSV
reports are saved under `attendance/reports/`; the dashboard also provides
download buttons. Back up the database and face images if you need to preserve
them. Face images and attendance records are sensitive personal data; restrict
access and collect them with appropriate consent.
