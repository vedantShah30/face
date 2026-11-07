
import sqlite3
from pathlib import Path


# --------------------------------------------------
# DATABASE CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DB_DIR = BASE_DIR / "attendance"
DB_PATH = DB_DIR / "attendance.db"

DEFAULT_COURSES = [
    "Digital",
    "Analog",
    "Control",
    "Communication"
]


# --------------------------------------------------
# ATTENDANCE DATABASE MANAGER
# --------------------------------------------------

class AttendanceManager:

    def __init__(self):

        DB_DIR.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(
            DB_PATH,
            check_same_thread=False
        )

        self.conn.row_factory = sqlite3.Row

        # Enable foreign-key enforcement
        self.conn.execute("PRAGMA foreign_keys = ON")

        self.create_tables()
        self.seed_courses()

        print("Attendance database initialized.")

    # --------------------------------------------------
    # CREATE ALL DATABASE TABLES
    # --------------------------------------------------

    def create_tables(self):

        cursor = self.conn.cursor()

        # 1. STUDENTS
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS students (
                student_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT
                    (datetime('now', 'localtime'))
            )
        """)

        # 2. COURSES
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS courses (
                course_id INTEGER PRIMARY KEY AUTOINCREMENT,
                course_name TEXT NOT NULL UNIQUE
            )
        """)

        # 3. ENROLLMENTS
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS enrollments (
                student_id TEXT NOT NULL,
                course_id INTEGER NOT NULL,

                enrolled_at TEXT NOT NULL DEFAULT
                    (datetime('now', 'localtime')),

                PRIMARY KEY (student_id, course_id),

                FOREIGN KEY (student_id)
                    REFERENCES students(student_id),

                FOREIGN KEY (course_id)
                    REFERENCES courses(course_id)
            )
        """)

        # 4. COURSE SCHEDULES
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS course_schedules (
                schedule_id INTEGER PRIMARY KEY AUTOINCREMENT,

                course_id INTEGER NOT NULL,

                day_of_week INTEGER NOT NULL
                    CHECK(day_of_week BETWEEN 0 AND 6),

                start_time TEXT NOT NULL,
                end_time TEXT NOT NULL,

                is_enabled INTEGER NOT NULL DEFAULT 1
                    CHECK(is_enabled IN (0, 1)),

                FOREIGN KEY (course_id)
                    REFERENCES courses(course_id),

                UNIQUE (
                    course_id,
                    day_of_week,
                    start_time,
                    end_time
                ),

                CHECK(start_time < end_time)
            )
        """)

        # 5. ATTENDANCE SESSIONS
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS attendance_sessions (
                session_id INTEGER PRIMARY KEY AUTOINCREMENT,

                course_id INTEGER NOT NULL,

                started_at TEXT NOT NULL,

                ended_at TEXT,

                status TEXT NOT NULL DEFAULT 'ACTIVE'
                    CHECK(status IN ('ACTIVE', 'COMPLETED')),

                FOREIGN KEY (course_id)
                    REFERENCES courses(course_id),

                CHECK(
                    (status = 'ACTIVE' AND ended_at IS NULL)
                    OR
                    (status = 'COMPLETED' AND ended_at IS NOT NULL)
                )
            )
        """)

        # Only one active session per course
        cursor.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS
                unique_active_session_per_course
            ON attendance_sessions(course_id)
            WHERE status = 'ACTIVE'
        """)

        # 6. ATTENDANCE RECORDS
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS attendance_records (
                attendance_id INTEGER PRIMARY KEY AUTOINCREMENT,

                session_id INTEGER NOT NULL,
                student_id TEXT NOT NULL,

                marked_at TEXT NOT NULL,

                status TEXT NOT NULL DEFAULT 'PRESENT'
                    CHECK(status = 'PRESENT'),

                FOREIGN KEY (session_id)
                    REFERENCES attendance_sessions(session_id),

                FOREIGN KEY (student_id)
                    REFERENCES students(student_id),

                UNIQUE(session_id, student_id)
            )
        """)

        self.conn.commit()

    # --------------------------------------------------
    # INSERT DEFAULT COURSES
    # --------------------------------------------------

    def seed_courses(self):

        cursor = self.conn.cursor()

        for course in DEFAULT_COURSES:

            cursor.execute("""
                INSERT OR IGNORE INTO courses (course_name)
                VALUES (?)
            """, (course,))

        self.conn.commit()

    # --------------------------------------------------
    # GET ALL COURSES
    # --------------------------------------------------

    def get_courses(self):

        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT course_id, course_name
            FROM courses
            ORDER BY course_id
        """)

        return cursor.fetchall()

    # --------------------------------------------------
    # GET ALL STUDENTS
    # --------------------------------------------------

    def get_students(self):

        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT student_id, name, created_at
            FROM students
            ORDER BY student_id
        """)

        return cursor.fetchall()
    
    
    # --------------------------------------------------
    # REGISTER STUDENT AND ENROL IN ALL COURSES
    # --------------------------------------------------

    def register_student(self, student_id, name):

        student_id = str(student_id).strip()
        name = name.strip()

        if not student_id:
            return False, "Student ID cannot be empty."

        if not name:
            return False, "Student name cannot be empty."

        try:

            with self.conn:

                cursor = self.conn.cursor()

                # Check that all default courses exist
                placeholders = ",".join(
                    ["?"] * len(DEFAULT_COURSES)
                )

                cursor.execute(f"""
                    SELECT course_id, course_name
                    FROM courses
                    WHERE course_name IN ({placeholders})
                """, DEFAULT_COURSES)

                courses = cursor.fetchall()

                if len(courses) != len(DEFAULT_COURSES):
                    return False, "Default courses are missing."

                # Register student
                cursor.execute("""
                    INSERT INTO students (student_id, name)
                    VALUES (?, ?)
                """, (student_id, name))

                # Automatically enrol student in all four courses
                enrolments = [
                    (student_id, course["course_id"])
                    for course in courses
                ]

                cursor.executemany("""
                    INSERT INTO enrollments (
                        student_id,
                        course_id
                    )
                    VALUES (?, ?)
                """, enrolments)

            return True, (
                f"{name} registered successfully "
                f"and enrolled in all {len(courses)} courses."
            )

        except sqlite3.IntegrityError:

            return False, (
                f"Student ID {student_id} already exists."
            )

        except sqlite3.Error as error:

            return False, f"Database error: {error}"

    # --------------------------------------------------
    # GET COURSES OF A PARTICULAR STUDENT
    # --------------------------------------------------

    def get_student_courses(self, student_id):

        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT
                courses.course_id,
                courses.course_name,
                enrollments.enrolled_at

            FROM enrollments

            JOIN courses
            ON enrollments.course_id = courses.course_id

            WHERE enrollments.student_id = ?

            ORDER BY courses.course_id
        """, (str(student_id),))

        return cursor.fetchall()

    # --------------------------------------------------
    # GET STUDENTS ENROLLED IN A COURSE
    # --------------------------------------------------

    def get_course_students(self, course_id):

        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT
                students.student_id,
                students.name,
                enrollments.enrolled_at

            FROM enrollments

            JOIN students
            ON enrollments.student_id = students.student_id

            WHERE enrollments.course_id = ?

            ORDER BY students.student_id
        """, (course_id,))

        return cursor.fetchall()


    # --------------------------------------------------
    # GET DATABASE TABLES
    # --------------------------------------------------

    def get_tables(self):

        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
            AND name NOT LIKE 'sqlite_%'
            ORDER BY name
        """)

        return [row["name"] for row in cursor.fetchall()]
    
    
    # --------------------------------------------------
    # ADD COURSE SCHEDULE
    # --------------------------------------------------

    def add_schedule(
        self,
        course_id,
        day_of_week,
        start_time,
        end_time
    ):

        from datetime import datetime

        # Validate weekday
        if not isinstance(day_of_week, int) or not 0 <= day_of_week <= 6:
            return False, "Invalid weekday. Use 0 to 6."

        # Validate time format
        try:
            start = datetime.strptime(start_time, "%H:%M")
            end = datetime.strptime(end_time, "%H:%M")
        except ValueError:
            return False, "Time must use HH:MM format."

        if start >= end:
            return False, "End time must be after start time."

        try:

            with self.conn:

                cursor = self.conn.cursor()

                # Check whether course exists
                cursor.execute("""
                    SELECT course_id
                    FROM courses
                    WHERE course_id = ?
                """, (course_id,))

                if cursor.fetchone() is None:
                    return False, "Course does not exist."

                # Insert schedule
                cursor.execute("""
                    INSERT INTO course_schedules (
                        course_id,
                        day_of_week,
                        start_time,
                        end_time,
                        is_enabled
                    )
                    VALUES (?, ?, ?, ?, 1)
                """, (
                    course_id,
                    day_of_week,
                    start_time,
                    end_time
                ))

            return True, "Schedule added successfully."

        except sqlite3.IntegrityError:
            return False, "This schedule already exists."

        except sqlite3.Error as error:
            return False, f"Database error: {error}"

    # --------------------------------------------------
    # GET ALL SCHEDULES
    # --------------------------------------------------

    def get_schedules(self, course_id=None):

        cursor = self.conn.cursor()

        query = """
            SELECT
                course_schedules.schedule_id,
                course_schedules.course_id,
                courses.course_name,
                course_schedules.day_of_week,
                course_schedules.start_time,
                course_schedules.end_time,
                course_schedules.is_enabled

            FROM course_schedules

            JOIN courses
            ON course_schedules.course_id = courses.course_id
        """

        params = ()

        if course_id is not None:
            query += " WHERE course_schedules.course_id = ?"
            params = (course_id,)

        query += """
            ORDER BY
                course_schedules.course_id,
                course_schedules.day_of_week,
                course_schedules.start_time
        """

        cursor.execute(query, params)

        return cursor.fetchall()

    # --------------------------------------------------
    # ENABLE / DISABLE SCHEDULE
    # --------------------------------------------------

    def set_schedule_status(self, schedule_id, enabled):

        if enabled not in (True, False, 0, 1):
            return False, "Invalid status."

        try:

            with self.conn:

                cursor = self.conn.cursor()

                cursor.execute("""
                    UPDATE course_schedules
                    SET is_enabled = ?
                    WHERE schedule_id = ?
                """, (
                    int(enabled),
                    schedule_id
                ))

                if cursor.rowcount == 0:
                    return False, "Schedule not found."

            return True, "Schedule status updated."

        except sqlite3.Error as error:
            return False, f"Database error: {error}"

    # --------------------------------------------------
    # DELETE SCHEDULE
    # --------------------------------------------------

    def delete_schedule(self, schedule_id):

        try:

            with self.conn:

                cursor = self.conn.cursor()

                cursor.execute("""
                    DELETE FROM course_schedules
                    WHERE schedule_id = ?
                """, (schedule_id,))

                if cursor.rowcount == 0:
                    return False, "Schedule not found."

            return True, "Schedule deleted successfully."

        except sqlite3.Error as error:
            return False, f"Database error: {error}"
        
    
    # --------------------------------------------------
    # START ATTENDANCE SESSION
    # --------------------------------------------------

    def start_attendance(self, course_id):

        from datetime import datetime

        try:

            with self.conn:

                cursor = self.conn.cursor()

                # Check whether course exists
                cursor.execute("""
                    SELECT course_name
                    FROM courses
                    WHERE course_id = ?
                """, (course_id,))

                course = cursor.fetchone()

                if course is None:
                    return False, "Course does not exist.", None

                # Check for an existing active session
                cursor.execute("""
                    SELECT session_id
                    FROM attendance_sessions
                    WHERE course_id = ?
                    AND status = 'ACTIVE'
                """, (course_id,))

                active_session = cursor.fetchone()

                if active_session:
                    return (
                        False,
                        "An attendance session is already active.",
                        active_session["session_id"]
                    )

                # Create new session
                started_at = datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )

                cursor.execute("""
                    INSERT INTO attendance_sessions (
                        course_id,
                        started_at,
                        status
                    )
                    VALUES (?, ?, 'ACTIVE')
                """, (
                    course_id,
                    started_at
                ))

                session_id = cursor.lastrowid

            return (
                True,
                f"Attendance started for {course['course_name']}.",
                session_id
            )

        except sqlite3.IntegrityError:
            return False, "An active session already exists.", None

        except sqlite3.Error as error:
            return False, f"Database error: {error}", None

    # --------------------------------------------------
    # END ATTENDANCE SESSION
    # --------------------------------------------------

    def end_attendance(self, session_id):

        from datetime import datetime

        try:

            with self.conn:

                cursor = self.conn.cursor()

                # Check whether session exists and is active
                cursor.execute("""
                    SELECT session_id
                    FROM attendance_sessions
                    WHERE session_id = ?
                    AND status = 'ACTIVE'
                """, (session_id,))

                session = cursor.fetchone()

                if session is None:
                    return False, (
                        "Session does not exist or is already completed."
                    )

                ended_at = datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )

                cursor.execute("""
                    UPDATE attendance_sessions
                    SET ended_at = ?,
                        status = 'COMPLETED'
                    WHERE session_id = ?
                    AND status = 'ACTIVE'
                """, (
                    ended_at,
                    session_id
                ))

            return True, "Attendance session ended successfully."

        except sqlite3.Error as error:
            return False, f"Database error: {error}"

    # --------------------------------------------------
    # GET ACTIVE SESSION FOR A COURSE
    # --------------------------------------------------

    def get_active_session_for_course(self, course_id):

        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT
                attendance_sessions.session_id,
                attendance_sessions.course_id,
                courses.course_name,
                attendance_sessions.started_at,
                attendance_sessions.status

            FROM attendance_sessions

            JOIN courses
            ON attendance_sessions.course_id = courses.course_id

            WHERE attendance_sessions.course_id = ?
            AND attendance_sessions.status = 'ACTIVE'
        """, (course_id,))

        return cursor.fetchone()

    # --------------------------------------------------
    # GET ALL ATTENDANCE SESSIONS
    # --------------------------------------------------

    def get_all_sessions(self, course_id=None):

        cursor = self.conn.cursor()

        query = """
            SELECT
                attendance_sessions.session_id,
                attendance_sessions.course_id,
                courses.course_name,
                attendance_sessions.started_at,
                attendance_sessions.ended_at,
                attendance_sessions.status

            FROM attendance_sessions

            JOIN courses
            ON attendance_sessions.course_id = courses.course_id
        """

        params = ()

        if course_id is not None:
            query += " WHERE attendance_sessions.course_id = ?"
            params = (course_id,)

        query += " ORDER BY attendance_sessions.session_id DESC"

        cursor.execute(query, params)

        return cursor.fetchall()

    
    # --------------------------------------------------
    # MARK STUDENT ATTENDANCE
    # --------------------------------------------------

    def mark_attendance(self, student_id, session_id):

        from datetime import datetime

        student_id = str(student_id).strip()

        if not student_id:
            return False, "Invalid student ID."

        try:

            with self.conn:

                cursor = self.conn.cursor()

                # --------------------------------------
                # 1. CHECK STUDENT
                # --------------------------------------

                cursor.execute("""
                    SELECT student_id, name
                    FROM students
                    WHERE student_id = ?
                """, (student_id,))

                student = cursor.fetchone()

                if student is None:
                    return False, "Student is not registered."

                # --------------------------------------
                # 2. CHECK ACTIVE SESSION
                # --------------------------------------

                cursor.execute("""
                    SELECT
                        session_id,
                        course_id,
                        status

                    FROM attendance_sessions

                    WHERE session_id = ?
                    AND status = 'ACTIVE'
                """, (session_id,))

                session = cursor.fetchone()

                if session is None:
                    return False, "Attendance session is not active."

                # --------------------------------------
                # 3. CHECK COURSE ENROLMENT
                # --------------------------------------

                cursor.execute("""
                    SELECT 1
                    FROM enrollments

                    WHERE student_id = ?
                    AND course_id = ?
                """, (
                    student_id,
                    session["course_id"]
                ))

                enrollment = cursor.fetchone()

                if enrollment is None:
                    return False, "Student is not enrolled in this course."

                # --------------------------------------
                # 4. CHECK DUPLICATE ATTENDANCE
                # --------------------------------------

                cursor.execute("""
                    SELECT attendance_id
                    FROM attendance_records

                    WHERE student_id = ?
                    AND session_id = ?
                """, (
                    student_id,
                    session_id
                ))

                existing_record = cursor.fetchone()

                if existing_record is not None:
                    return False, "Attendance already marked."

                # --------------------------------------
                # 5. INSERT ATTENDANCE
                # --------------------------------------

                marked_at = datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )

                cursor.execute("""
                    INSERT INTO attendance_records (
                        session_id,
                        student_id,
                        marked_at,
                        status
                    )
                    VALUES (?, ?, ?, 'PRESENT')
                """, (
                    session_id,
                    student_id,
                    marked_at
                ))

                return True, (
                    f"Attendance marked for {student['name']}."
                )

        except sqlite3.IntegrityError:
            return False, "Attendance already marked."

        except sqlite3.Error as error:
            return False, f"Database error: {error}"

    # --------------------------------------------------
    # GET ATTENDANCE FOR A SESSION
    # --------------------------------------------------

    def get_session_attendance(self, session_id):

        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT
                students.student_id,
                students.name,
                attendance_records.marked_at,
                attendance_records.status

            FROM attendance_records

            JOIN students
            ON attendance_records.student_id = students.student_id

            WHERE attendance_records.session_id = ?

            ORDER BY attendance_records.marked_at
        """, (session_id,))

        return cursor.fetchall()

    # --------------------------------------------------
    # GET ATTENDANCE COUNT FOR A SESSION
    # --------------------------------------------------

    def get_attendance_count(self, session_id):

        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT COUNT(*) AS total_present

            FROM attendance_records

            WHERE session_id = ?
            AND status = 'PRESENT'
        """, (session_id,))

        return cursor.fetchone()["total_present"]
    
    
    # --------------------------------------------------
    # STUDENT-WISE COURSE ATTENDANCE SUMMARY
    # --------------------------------------------------

    def get_course_attendance_summary(self, course_id):

        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT
                students.student_id,
                students.name,

                COUNT(DISTINCT sessions.session_id)
                    AS total_classes,

                COUNT(DISTINCT
                    CASE
                        WHEN attendance_records.status = 'PRESENT'
                        THEN sessions.session_id
                    END
                ) AS classes_attended,

                CASE
                    WHEN COUNT(DISTINCT sessions.session_id) = 0
                    THEN 0

                    ELSE ROUND(
                        100.0 *
                        COUNT(DISTINCT
                            CASE
                                WHEN attendance_records.status = 'PRESENT'
                                THEN sessions.session_id
                            END
                        )
                        /
                        COUNT(DISTINCT sessions.session_id),
                        2
                    )
                END AS attendance_percentage

            FROM enrollments

            JOIN students
                ON enrollments.student_id = students.student_id

            LEFT JOIN attendance_sessions AS sessions
                ON sessions.course_id = enrollments.course_id
                AND sessions.status = 'COMPLETED'

            LEFT JOIN attendance_records
                ON attendance_records.session_id = sessions.session_id
                AND attendance_records.student_id = students.student_id

            WHERE enrollments.course_id = ?

            GROUP BY
                students.student_id,
                students.name

            ORDER BY students.student_id
        """, (course_id,))

        columns = [description[0] for description in cursor.description]

        return [
            dict(zip(columns, row))
            for row in cursor.fetchall()
        ]

    # --------------------------------------------------
    # DAILY ATTENDANCE REPORT
    # --------------------------------------------------

    def get_course_daily_report(self, course_id):

        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT
                DATE(sessions.started_at) AS attendance_date,

                courses.course_name,

                COUNT(DISTINCT sessions.session_id)
                    AS sessions_held,

                COUNT(DISTINCT attendance_records.student_id)
                    AS students_present,

                (
                    SELECT COUNT(*)
                    FROM enrollments
                    WHERE enrollments.course_id = sessions.course_id
                ) AS total_enrolled

            FROM attendance_sessions AS sessions

            JOIN courses
                ON sessions.course_id = courses.course_id

            LEFT JOIN attendance_records
                ON attendance_records.session_id = sessions.session_id
                AND attendance_records.status = 'PRESENT'

            WHERE sessions.course_id = ?
            AND sessions.status = 'COMPLETED'

            GROUP BY
                DATE(sessions.started_at),
                courses.course_name

            ORDER BY attendance_date
        """, (course_id,))

        return cursor.fetchall()

    # --------------------------------------------------
    # DETAILED ATTENDANCE REPORT
    # Includes PRESENT and ABSENT students
    # --------------------------------------------------

    def get_course_detailed_report(self, course_id):

        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT
                sessions.session_id,

                DATE(sessions.started_at) AS attendance_date,

                sessions.started_at,
                sessions.ended_at,

                students.student_id,
                students.name,

                CASE
                    WHEN attendance_records.attendance_id IS NOT NULL
                    THEN 'PRESENT'
                    ELSE 'ABSENT'
                END AS attendance_status,

                attendance_records.marked_at

            FROM attendance_sessions AS sessions

            JOIN enrollments
                ON enrollments.course_id = sessions.course_id

            JOIN students
                ON students.student_id = enrollments.student_id

            LEFT JOIN attendance_records
                ON attendance_records.session_id = sessions.session_id
                AND attendance_records.student_id = students.student_id

            WHERE sessions.course_id = ?
            AND sessions.status = 'COMPLETED'

            ORDER BY
                sessions.started_at,
                students.student_id
        """, (course_id,))

        return cursor.fetchall()

    # --------------------------------------------------
    # GET INDIVIDUAL STUDENT'S COURSE HISTORY
    # --------------------------------------------------

    def get_student_course_history(self, student_id, course_id):

        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT
                sessions.session_id,

                sessions.started_at,
                sessions.ended_at,

                CASE
                    WHEN attendance_records.attendance_id IS NOT NULL
                    THEN 'PRESENT'
                    ELSE 'ABSENT'
                END AS attendance_status,

                attendance_records.marked_at

            FROM attendance_sessions AS sessions

            JOIN enrollments
                ON enrollments.course_id = sessions.course_id

            LEFT JOIN attendance_records
                ON attendance_records.session_id = sessions.session_id
                AND attendance_records.student_id = enrollments.student_id

            WHERE enrollments.student_id = ?
            AND sessions.course_id = ?
            AND sessions.status = 'COMPLETED'

            ORDER BY sessions.started_at
        """, (str(student_id), course_id))

        return cursor.fetchall()

    # --------------------------------------------------
    # EXPORT COURSE REPORTS AS CSV
    # --------------------------------------------------

    def export_course_reports(self, course_id, output_dir=None):

        import csv
        from pathlib import Path

        # Get course name
        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT course_name
            FROM courses
            WHERE course_id = ?
        """, (course_id,))

        course = cursor.fetchone()

        if course is None:
            return False, "Course does not exist."

        course_name = course["course_name"]

        # Create report directory
        if output_dir is None:

            report_dir = DB_DIR / "reports"

        else:

            report_dir = Path(output_dir)

        report_dir.mkdir(parents=True, exist_ok=True)

        filename = course_name.lower().replace(" ", "_")

        # Prepare report datasets
        reports = {

            "student_summary": (
                self.get_course_attendance_summary(course_id),
                report_dir / f"{filename}_student_summary.csv"
            ),

            "daily_report": (
                self.get_course_daily_report(course_id),
                report_dir / f"{filename}_daily_report.csv"
            ),

            "detailed_report": (
                self.get_course_detailed_report(course_id),
                report_dir / f"{filename}_detailed_report.csv"
            )
        }

        generated_files = {}

        try:

            for report_name, (rows, file_path) in reports.items():

                # Convert SQLite rows into dictionaries
                records = [
                    dict(row)
                    for row in rows
                ]

                # Write CSV
                with open(
                    file_path,
                    "w",
                    newline="",
                    encoding="utf-8-sig"
                ) as file:

                    if records:

                        writer = csv.DictWriter(
                            file,
                            fieldnames=records[0].keys()
                        )

                        writer.writeheader()
                        writer.writerows(records)

                    else:

                        # Empty report still gets a valid CSV file
                        file.write("No records available\n")

                generated_files[report_name] = str(file_path)

            return True, generated_files

        except (OSError, csv.Error) as error:

            return False, f"Export failed: {error}"


    # --------------------------------------------------
    # CLOSE DATABASE CONNECTION
    # --------------------------------------------------

    def close(self):

        if self.conn:
            self.conn.close()


# --------------------------------------------------
# INITIALIZATION TEST
# --------------------------------------------------

if __name__ == "__main__":

    db = AttendanceManager()

    print("\nDATABASE LOCATION:")
    print(DB_PATH)

    print("\nCREATED TABLES:")

    for table in db.get_tables():
        print("-", table)

    print("\nREGISTERED COURSES:")

    for course in db.get_courses():

        print(
            course["course_id"],
            "-",
            course["course_name"]
        )

    print("\nREGISTERED STUDENTS:")

    students = db.get_students()

    if not students:
        print("No students registered yet.")

    else:

        for student in students:

            print(
                student["student_id"],
                "-",
                student["name"]
            )

    db.close()

    print("\nDatabase initialization test completed.")
