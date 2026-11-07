
import streamlit as st
import pandas as pd
import subprocess
import sys
import os

from pathlib import Path
from datetime import time

from attendance_manager import AttendanceManager, DB_DIR


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Campus Attendance",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================================
# THEME AND CUSTOM CSS
# ==========================================================

st.markdown("""
<style>

/* Main background */
.stApp {
    background-color: #FEFAE0;
    color: #30382B;
}

/* Main content */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #E9EDC9;
    border-right: 1px solid #CCD5AE;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] p {
    color: #30382B;
}

/* Headings */
h1, h2, h3 {
    color: #30382B !important;
    font-weight: 650 !important;
}

/* Body text */
p, label, span {
    color: #3E4638;
}

/* Buttons */
.stButton > button {
    background-color: #D4A373;
    color: #FFFFFF;
    border: none;
    border-radius: 9px;
    padding: 0.55rem 1.1rem;
    font-weight: 600;
    transition: 0.2s;
}

.stButton > button:hover {
    background-color: #B98556;
    color: #FFFFFF;
    border: none;
}

/* Form submit buttons */
.stFormSubmitButton > button {
    background-color: #59634B;
    color: white;
    border-radius: 9px;
    border: none;
}

.stFormSubmitButton > button:hover {
    background-color: #414A36;
    color: white;
}

/* Input fields */
.stTextInput input,
.stSelectbox div[data-baseweb="select"],
.stNumberInput input {
    background-color: #FFFEF7;
    border-radius: 8px;
}

/* Metric cards */
[data-testid="stMetric"] {
    background-color: #FFFEF7;
    border: 1px solid #CCD5AE;
    padding: 18px;
    border-radius: 12px;
}

/* Dataframes */
[data-testid="stDataFrame"] {
    border: 1px solid #CCD5AE;
    border-radius: 10px;
}

/* Dividers */
hr {
    border-color: #CCD5AE;
}

/* Alerts */
div[data-testid="stAlert"] {
    border-radius: 10px;
}

/* Radio navigation */
div[role="radiogroup"] {
    gap: 10px;
}

/* Sidebar brand */
.brand-title {
    font-size: 24px;
    font-weight: 750;
    color: #30382B;
    margin-bottom: 3px;
}

.brand-subtitle {
    font-size: 12px;
    color: #69735C;
    margin-bottom: 25px;
}

/* Section heading */
.section-caption {
    color: #78816B;
    font-size: 13px;
    margin-top: -12px;
    margin-bottom: 20px;
}

/* Custom cards */
.info-card {
    background: #FFFEF7;
    border: 1px solid #CCD5AE;
    padding: 20px;
    border-radius: 12px;
    min-height: 110px;
}

.info-card-label {
    font-size: 12px;
    color: #727C64;
    font-weight: 600;
    text-transform: uppercase;
}

.info-card-value {
    font-size: 28px;
    font-weight: 750;
    color: #30382B;
    margin-top: 8px;
}

.info-card-note {
    font-size: 12px;
    color: #78816B;
}

/* Hide Streamlit branding */
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# ==========================================================
# DATABASE CONNECTION
# ==========================================================

db = AttendanceManager()

PROJECT_DIR = Path(__file__).resolve().parent
LIVE_SCRIPT = PROJECT_DIR / "attendance_live.py"


# ==========================================================
# HELPER FUNCTIONS
# ==========================================================

DAY_NAMES = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]


def rows_to_df(rows):

    return pd.DataFrame([
        dict(row) for row in rows
    ])


def get_course_map():

    return {
        course["course_name"]: course["course_id"]
        for course in db.get_courses()
    }


def get_course_name(course_id):

    for course in db.get_courses():

        if course["course_id"] == course_id:
            return course["course_name"]

    return "Unknown course"


def format_time(value):

    return value.strftime("%H:%M")


def get_active_process():

    process_info = st.session_state.get(
        "attendance_process"
    )

    if process_info is None:
        return None

    process = process_info["process"]

    if process.poll() is not None:

        st.session_state.attendance_process = None

        return None

    return process_info


# ==========================================================
# SIDEBAR NAVIGATION
# ==========================================================

with st.sidebar:

    st.markdown(
        '<div class="brand-title">Campus Attendance</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="brand-subtitle">'
        'Student Management System'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()

    selected_page = st.radio(
        "NAVIGATION",
        [
            "Dashboard",
            "Students",
            "Schedules",
            "Live Attendance",
            "Reports"
        ],
        label_visibility="visible"
    )

    st.divider()

    st.markdown("**SYSTEM INFORMATION**")

    st.caption("Database: SQLite")

    st.caption("Recognition: ArcFace + SVM")

    st.caption("Courses: 4")

    st.markdown("---")

    st.caption("Campus Attendance v1.0")


# ==========================================================
# DASHBOARD PAGE
# ==========================================================

if selected_page == "Dashboard":

    st.title("Dashboard")

    st.markdown(
        '<div class="section-caption">'
        'Overview of your attendance management system'
        '</div>',
        unsafe_allow_html=True
    )

    students = db.get_students()

    courses = db.get_courses()

    sessions = db.get_all_sessions()

    active_sessions = [
        session for session in sessions
        if session["status"] == "ACTIVE"
    ]

    completed_sessions = [
        session for session in sessions
        if session["status"] == "COMPLETED"
    ]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Students",
        len(students)
    )

    col2.metric(
        "Total Courses",
        len(courses)
    )

    col3.metric(
        "Completed Classes",
        len(completed_sessions)
    )

    col4.metric(
        "Active Sessions",
        len(active_sessions)
    )

    st.divider()

    left, right = st.columns([1.3, 1])

    with left:

        st.subheader("Course Overview")

        course_data = []

        for course in courses:

            enrolled = db.get_course_students(
                course["course_id"]
            )

            course_sessions = [
                session for session in completed_sessions
                if session["course_id"] == course["course_id"]
            ]

            course_data.append({

                "Course": course["course_name"],

                "Enrolled Students": len(enrolled),

                "Completed Classes": len(course_sessions)

            })

        st.dataframe(
            pd.DataFrame(course_data),
            use_container_width=True,
            hide_index=True
        )

    with right:

        st.subheader("Active Sessions")

        if active_sessions:

            for session in active_sessions:

                st.info(
                    f"**{session['course_name']}**\n\n"
                    f"Session ID: {session['session_id']}\n\n"
                    f"Started: {session['started_at']}"
                )

        else:

            st.success("No active attendance sessions.")

    st.divider()

    st.subheader("Recent Attendance Sessions")

    if sessions:

        recent = rows_to_df(sessions[:10])

        st.dataframe(
            recent,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info("No attendance sessions have been created yet.")


# ==========================================================
# STUDENTS PAGE
# ==========================================================

elif selected_page == "Students":

    st.title("Student Management")

    st.markdown(
        '<div class="section-caption">'
        'Register students and manage their course enrolments'
        '</div>',
        unsafe_allow_html=True
    )

    left, right = st.columns([1, 1.4])

    with left:

        st.subheader("Register New Student")

        with st.form("student_registration_form"):

            student_id = st.text_input(
                "Roll Number",
                placeholder="Example: 230108060"
            )

            student_name = st.text_input(
                "Student Name",
                placeholder="Enter full name"
            )

            submitted = st.form_submit_button(
                "Register Student",
                use_container_width=True
            )

            if submitted:

                success, message = db.register_student(
                    student_id,
                    student_name
                )

                if success:
                    st.success(message)

                else:
                    st.error(message)

        st.warning(
            "This registers the student in the attendance "
            "database and enrols them in all four courses. "
            "Face dataset capture and model retraining are "
            "still separate operations."
        )

    with right:

        st.subheader("Registration Summary")

        students = db.get_students()

        st.metric(
            "Registered Students",
            len(students)
        )

        if students:

            st.success(
                f"{len(students)} students are currently registered."
            )

    st.divider()

    st.subheader("Registered Students")

    students = db.get_students()

    if students:

        student_data = []

        for student in students:

            enrolled_courses = db.get_student_courses(
                student["student_id"]
            )

            course_names = ", ".join(
                course["course_name"]
                for course in enrolled_courses
            )

            student_data.append({

                "Roll Number": student["student_id"],

                "Name": student["name"],

                "Enrolled Courses": course_names,

                "Registered At": student["created_at"]

            })

        st.dataframe(
            pd.DataFrame(student_data),
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info("No students registered yet.")


# ==========================================================
# SCHEDULES PAGE
# ==========================================================

elif selected_page == "Schedules":

    st.title("Course Schedules")

    st.markdown(
        '<div class="section-caption">'
        'Configure weekly attendance schedules for each course'
        '</div>',
        unsafe_allow_html=True
    )

    course_map = get_course_map()

    with st.container(border=True):

        st.subheader("Add New Schedule")

        with st.form("schedule_form"):

            course_name = st.selectbox(
                "Select Course",
                list(course_map.keys())
            )

            selected_days = st.multiselect(
                "Select Days",
                DAY_NAMES,
                default=[]
            )

            col1, col2 = st.columns(2)

            with col1:

                start_time = st.time_input(
                    "Start Time",
                    value=time(9, 0)
                )

            with col2:

                end_time = st.time_input(
                    "End Time",
                    value=time(10, 0)
                )

            submitted = st.form_submit_button(
                "Save Schedule",
                use_container_width=True
            )

            if submitted:

                if not selected_days:

                    st.error("Select at least one day.")

                elif start_time >= end_time:

                    st.error("End time must be after start time.")

                else:

                    course_id = course_map[course_name]

                    success_count = 0
                    messages = []

                    for day in selected_days:

                        day_number = DAY_NAMES.index(day)

                        success, message = db.add_schedule(
                            course_id,
                            day_number,
                            format_time(start_time),
                            format_time(end_time)
                        )

                        if success:
                            success_count += 1

                        messages.append(
                            f"{day}: {message}"
                        )

                    if success_count:

                        st.success(
                            f"Added {success_count} schedule(s)."
                        )

                    for message in messages:

                        st.caption(message)

    st.divider()

    st.subheader("Existing Schedules")

    schedules = db.get_schedules()

    if schedules:

        schedule_data = []

        for schedule in schedules:

            schedule_data.append({

                "Schedule ID": schedule["schedule_id"],

                "Course": schedule["course_name"],

                "Day": DAY_NAMES[schedule["day_of_week"]],

                "Start": schedule["start_time"],

                "End": schedule["end_time"],

                "Status": (
                    "Enabled"
                    if schedule["is_enabled"]
                    else "Disabled"
                )

            })

        st.dataframe(
            pd.DataFrame(schedule_data),
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        st.subheader("Manage Schedule")

        schedule_options = {
            schedule["schedule_id"]: schedule
            for schedule in schedules
        }

        selected_schedule_id = st.selectbox(
            "Select Schedule",
            list(schedule_options.keys()),
            format_func=lambda sid: (
                f"{schedule_options[sid]['course_name']} | "
                f"{DAY_NAMES[schedule_options[sid]['day_of_week']]} | "
                f"{schedule_options[sid]['start_time']}-"
                f"{schedule_options[sid]['end_time']}"
            )
        )

        selected_schedule = schedule_options[
            selected_schedule_id
        ]

        col1, col2 = st.columns(2)

        with col1:

            if selected_schedule["is_enabled"]:

                if st.button(
                    "Disable Schedule",
                    use_container_width=True
                ):

                    success, message = db.set_schedule_status(
                        selected_schedule_id,
                        False
                    )

                    if success:
                        st.success(message)
                        st.rerun()

                    else:
                        st.error(message)

            else:

                if st.button(
                    "Enable Schedule",
                    use_container_width=True
                ):

                    success, message = db.set_schedule_status(
                        selected_schedule_id,
                        True
                    )

                    if success:
                        st.success(message)
                        st.rerun()

                    else:
                        st.error(message)

        with col2:

            if st.button(
                "Delete Schedule",
                type="secondary",
                use_container_width=True
            ):

                success, message = db.delete_schedule(
                    selected_schedule_id
                )

                if success:
                    st.success(message)
                    st.rerun()

                else:
                    st.error(message)

    else:

        st.info("No schedules have been configured.")


# ==========================================================
# LIVE ATTENDANCE PAGE
# ==========================================================

elif selected_page == "Live Attendance":

    st.title("Live Attendance")

    st.markdown(
        '<div class="section-caption">'
        'Start and manage live face-recognition attendance sessions'
        '</div>',
        unsafe_allow_html=True
    )

    process_info = get_active_process()

    courses = db.get_courses()

    course_map = {
        course["course_name"]: course["course_id"]
        for course in courses
    }

    st.subheader("Attendance Control")

    if process_info is None:

        selected_course = st.selectbox(
            "Select Course",
            list(course_map.keys())
        )

        course_id = course_map[selected_course]

        active_session = db.get_active_session_for_course(
            course_id
        )

        if active_session:

            st.warning(
                f"An active session already exists for "
                f"{selected_course}. Session ID: "
                f"{active_session['session_id']}."
            )

            st.caption(
                "If a previous webcam process was interrupted, "
                "resolve that session before starting another."
            )

        else:

            st.info(
                "Select a course and start attendance. "
                "The webcam will open in a separate window."
            )

            if st.button(
                "Start Attendance",
                use_container_width=True
            ):

                if not LIVE_SCRIPT.exists():

                    st.error(
                        "attendance_live.py was not found."
                    )

                else:

                    control_dir = DB_DIR / "controls"

                    control_dir.mkdir(
                        parents=True,
                        exist_ok=True
                    )

                    stop_file = (
                        control_dir /
                        f"stop_course_{course_id}.flag"
                    )

                    if stop_file.exists():
                        stop_file.unlink()

                    try:

                        process = subprocess.Popen(
                            [
                                sys.executable,
                                str(LIVE_SCRIPT),
                                "--course-id",
                                str(course_id),
                                "--stop-file",
                                str(stop_file)
                            ],
                            cwd=str(PROJECT_DIR)
                        )

                        st.session_state.attendance_process = {

                            "process": process,

                            "course_id": course_id,

                            "course_name": selected_course,

                            "stop_file": str(stop_file)

                        }

                        st.success(
                            f"Launching attendance for {selected_course}."
                        )

                    except Exception as error:

                        st.error(
                            f"Could not start attendance: {error}"
                        )

    else:

        st.success("Attendance process is running.")

        st.markdown(
            f"### {process_info['course_name']}"
        )

        st.caption(
            f"Course ID: {process_info['course_id']}"
        )

        active_session = db.get_active_session_for_course(
            process_info["course_id"]
        )

        if active_session:

            st.metric(
                "Active Session ID",
                active_session["session_id"]
            )

            st.caption(
                f"Started at: {active_session['started_at']}"
            )

        else:

            st.info(
                "The webcam process is starting or initializing."
            )

        st.warning(
            "The webcam is displayed in a separate OpenCV window. "
            "Keep that window open while attendance is running."
        )

        col1, col2 = st.columns(2)

        with col1:

            if st.button(
                "End Attendance",
                type="primary",
                use_container_width=True
            ):

                stop_file = Path(
                    process_info["stop_file"]
                )

                stop_file.parent.mkdir(
                    parents=True,
                    exist_ok=True
                )

                stop_file.write_text(
                    "STOP",
                    encoding="utf-8"
                )

                st.info(
                    "Stop requested. The webcam process will "
                    "finish and close the session."
                )

        with col2:

            if st.button(
                "Refresh Status",
                use_container_width=True
            ):

                st.rerun()

    st.divider()

    st.subheader("Current Active Sessions")

    all_sessions = db.get_all_sessions()

    active_sessions = [
        session for session in all_sessions
        if session["status"] == "ACTIVE"
    ]

    if active_sessions:

        st.dataframe(
            rows_to_df(active_sessions),
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info("There are currently no active sessions.")


# ==========================================================
# REPORTS PAGE
# ==========================================================

elif selected_page == "Reports":

    st.title("Attendance Reports")

    st.markdown(
        '<div class="section-caption">'
        'Analyze attendance and export course-wise reports'
        '</div>',
        unsafe_allow_html=True
    )

    course_map = get_course_map()

    selected_course = st.selectbox(
        "Select Course",
        list(course_map.keys())
    )

    course_id = course_map[selected_course]

    st.divider()

    # ------------------------------------------------------
    # STUDENT SUMMARY
    # ------------------------------------------------------

    st.subheader("Student Attendance Summary")

    summary = db.get_course_attendance_summary(
        course_id
    )

    if summary:

        summary_df = rows_to_df(summary)

        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True
        )

        if len(summary_df):

            st.bar_chart(
                summary_df.set_index("name")[
                    "attendance_percentage"
                ]
            )

    else:

        st.info("No student attendance data available.")

    st.divider()

    # ------------------------------------------------------
    # DAILY REPORT
    # ------------------------------------------------------

    st.subheader("Daily Attendance")

    daily_report = db.get_course_daily_report(
        course_id
    )

    if daily_report:

        daily_df = rows_to_df(daily_report)

        st.dataframe(
            daily_df,
            use_container_width=True,
            hide_index=True
        )

        st.bar_chart(
            daily_df.set_index("attendance_date")[
                "students_present"
            ]
        )

    else:

        st.info("No completed sessions available.")

    st.divider()

    # ------------------------------------------------------
    # DETAILED REPORT
    # ------------------------------------------------------

    st.subheader("Detailed Attendance History")

    detailed_report = db.get_course_detailed_report(
        course_id
    )

    if detailed_report:

        st.dataframe(
            rows_to_df(detailed_report),
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info("No detailed attendance records available.")

    st.divider()

    # ------------------------------------------------------
    # EXPORT REPORTS
    # ------------------------------------------------------

    st.subheader("Download Reports")

    if st.button(
        "Generate CSV Reports",
        use_container_width=True
    ):

        success, result = db.export_course_reports(
            course_id
        )

        if success:

            st.session_state.report_files = result

            st.success(
                f"Reports generated for {selected_course}."
            )

        else:

            st.error(result)

    report_files = st.session_state.get(
        "report_files",
        {}
    )

    if report_files:

        download_columns = st.columns(3)

        report_labels = {

            "student_summary": "Student Summary",

            "daily_report": "Daily Report",

            "detailed_report": "Detailed Report"

        }

        for index, (report_name, file_path) in enumerate(
            report_files.items()
        ):

            path = Path(file_path)

            if path.exists():

                with download_columns[index]:

                    st.download_button(
                        label=f"Download {report_labels[report_name]}",
                        data=path.read_bytes(),
                        file_name=path.name,
                        mime="text/csv",
                        use_container_width=True
                    )


# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.markdown(
    """
    <div style="
        text-align:center;
        color:#78816B;
        font-size:12px;
        padding:10px;
    ">
        Campus Attendance Management System · Version 1.0
    </div>
    """,
    unsafe_allow_html=True
)


# ==========================================================
# CLOSE DATABASE CONNECTION
# ==========================================================

db.close()
