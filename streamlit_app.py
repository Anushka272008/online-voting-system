import os
import io
import json
import sqlite3

import cv2
import numpy as np
import streamlit as st

from face_utils import get_face_feature, compare_faces

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(BASE_DIR, "database", "voting.db")
SYMBOL_DIR = os.path.join(BASE_DIR, "static", "images")

# SFace's commonly used cosine similarity threshold.
# Higher = stricter face verification.
FACE_THRESHOLD = 0.363

st.set_page_config(
    page_title="Online Voting System",
    page_icon="🗳️",
    layout="centered"
)

def db():
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    con = sqlite3.connect(DB, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con

def initialize_database():
    con = db()
    c = con.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS voters(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            enrollment TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            contact TEXT UNIQUE NOT NULL,
            face_embedding TEXT,
            has_voted INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS candidates(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            party TEXT,
            symbol TEXT
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS votes(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            voter_id INTEGER UNIQUE NOT NULL,
            candidate_id INTEGER NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(voter_id) REFERENCES voters(id),
            FOREIGN KEY(candidate_id) REFERENCES candidates(id)
        )
    """)

    candidates = [
        ("Rahul", "Yuva Shakti Party", "symbol1.jpeg"),
        ("Sneha", "The New Era Party", "symbol2.jpeg"),
        ("Priya", "Students Unity Party", "symbol3.jpeg"),
        ("Ajay", "Nayi Soch Party", "symbol4.jpeg")
    ]

    if c.execute("SELECT COUNT(*) FROM candidates").fetchone()[0] == 0:
        c.executemany(
            "INSERT INTO candidates(name,party,symbol) VALUES(?,?,?)",
            candidates
        )

    con.commit()
    con.close()

def image_to_frame(uploaded):
    data = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

def verify_captured_face(uploaded, saved_embedding=None):
    frame = image_to_frame(uploaded)
    if frame is None:
        return None, None, "Could not read the camera image."

    feature = get_face_feature(frame)

    if feature is None:
        return None, frame, "Exactly one face must be visible."

    if saved_embedding is None:
        return feature, frame, "Face detected."

    score = compare_faces(feature, saved_embedding)
    if score >= FACE_THRESHOLD:
        return score, frame, f"Face verified. Similarity: {score:.3f}"
    return score, frame, f"Face not matched. Similarity: {score:.3f}"

def home():
    st.title("🗳️ Online Voting System")
    st.caption("Secure voting with face verification")

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        if st.button("👤 Voter Registration", use_container_width=True):
            st.session_state.page = "register"
            st.rerun()
    with c2:
        if st.button("🔐 Voter Login", use_container_width=True):
            st.session_state.page = "login"
            st.rerun()

    if st.button("⚙️ Admin Login", use_container_width=True):
        st.session_state.page = "admin_login"
        st.rerun()

def register():
    st.header("Voter Registration")

    enrollment = st.text_input("Enrollment Number")
    name = st.text_input("Name")
    contact = st.text_input("Email / Phone")

    st.info("Allow camera access, position one face clearly in front of the camera, then capture the image.")

    photo = st.camera_input("Capture your face")

    if st.button("Register", type="primary", use_container_width=True):
        if not enrollment or not name or not contact:
            st.error("Please fill all fields.")
            return
        if photo is None:
            st.error("Please capture a face image.")
            return

        feature, frame, message = verify_captured_face(photo)

        if feature is None:
            st.error(message)
            if frame is not None:
                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption=message)
            return

        con = db()
        try:
            con.execute(
                """INSERT INTO voters
                   (enrollment,name,contact,face_embedding)
                   VALUES(?,?,?,?)""",
                (enrollment.strip(), name.strip(), contact.strip(),
                 json.dumps(feature.tolist()))
            )
            con.commit()
            st.success("Registration successful.")
        except sqlite3.IntegrityError:
            st.error("Enrollment number or contact already exists.")
        finally:
            con.close()

    if st.button("← Back to Home"):
        st.session_state.page = "home"
        st.rerun()

def voter_login():
    st.header("Voter Login")

    enrollment = st.text_input("Enrollment Number")
    photo = st.camera_input("Capture your face")

    if st.button("Verify & Login", type="primary", use_container_width=True):
        if not enrollment:
            st.error("Enter your enrollment number.")
            return
        if photo is None:
            st.error("Capture your face.")
            return

        con = db()
        voter = con.execute(
            "SELECT * FROM voters WHERE enrollment=?",
            (enrollment.strip(),)
        ).fetchone()
        con.close()

        if not voter:
            st.error("Voter not found.")
            return

        if voter["has_voted"]:
            st.error("You have already voted.")
            return

        if not voter["face_embedding"]:
            st.error("Face enrollment not found.")
            return

        saved = np.asarray(json.loads(voter["face_embedding"]), dtype=np.float32)
        score, frame, message = verify_captured_face(photo, saved)

        if frame is not None:
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption=message)

        if score is None:
            st.error(message)
            return

        if score >= FACE_THRESHOLD:
            st.session_state.voter = enrollment.strip()
            st.success("Face verified successfully.")
            st.session_state.page = "vote"
            st.rerun()
        else:
            st.error(f"Face verification failed. Similarity: {score:.3f}")

    if st.button("← Back to Home"):
        st.session_state.page = "home"
        st.rerun()

def vote_page():
    if "voter" not in st.session_state:
        st.session_state.page = "login"
        st.rerun()

    st.title("CAST YOUR VOTE")

    con = db()
    voter = con.execute(
        "SELECT * FROM voters WHERE enrollment=?",
        (st.session_state.voter,)
    ).fetchone()

    if not voter or voter["has_voted"]:
        con.close()
        st.error("You are not eligible to vote.")
        st.session_state.pop("voter", None)
        if st.button("Back to Home"):
            st.session_state.page = "home"
            st.rerun()
        return

    candidates = con.execute(
        "SELECT id,name,party,symbol FROM candidates ORDER BY id"
    ).fetchall()
    con.close()

    for c in candidates:
        with st.container(border=True):
            left, middle, right = st.columns([1, 3, 1])

            symbol_path = os.path.join(SYMBOL_DIR, c["symbol"])
            with left:
                if os.path.exists(symbol_path):
                    st.image(symbol_path, width=80)

            with middle:
                st.subheader(c["name"])
                st.write(c["party"])

            with right:
                if st.button("VOTE", key=f"vote_{c['id']}"):
                    st.session_state.selected_candidate = c["id"]
                    st.session_state.selected_name = c["name"]

    if "selected_candidate" in st.session_state:
        st.warning(
            f"You selected **{st.session_state.selected_name}**. "
            "Voting is final and cannot be changed."
        )
        a, b = st.columns(2)
        with a:
            if st.button("Confirm Vote", type="primary", use_container_width=True):
                cast_vote(st.session_state.selected_candidate)
        with b:
            if st.button("Cancel Selection", use_container_width=True):
                st.session_state.pop("selected_candidate", None)
                st.session_state.pop("selected_name", None)
                st.rerun()

def cast_vote(candidate_id):
    con = db()
    voter = con.execute(
        "SELECT id,has_voted FROM voters WHERE enrollment=?",
        (st.session_state.get("voter"),)
    ).fetchone()

    if not voter or voter["has_voted"]:
        con.close()
        st.error("Already voted or voter session expired.")
        return

    try:
        con.execute(
            "INSERT INTO votes(voter_id,candidate_id) VALUES(?,?)",
            (voter["id"], candidate_id)
        )
        con.execute(
            "UPDATE voters SET has_voted=1 WHERE id=?",
            (voter["id"],)
        )
        con.commit()
    except sqlite3.IntegrityError:
        con.rollback()
        st.error("Vote could not be submitted. The voter may have already voted.")
        con.close()
        return

    con.close()
    st.session_state.clear()
    st.session_state.page = "home"
    st.success("Vote submitted successfully.")
    st.rerun()

def admin_login():
    st.header("Admin Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login", type="primary", use_container_width=True):
        admin_user = st.secrets.get("ADMIN_USERNAME", "admin")
        admin_pass = st.secrets.get("ADMIN_PASSWORD", "admin123")

        if username == admin_user and password == admin_pass:
            st.session_state.admin = True
            st.session_state.page = "admin"
            st.rerun()
        else:
            st.error("Invalid admin login.")

    if st.button("← Back to Home"):
        st.session_state.page = "home"
        st.rerun()

def admin_dashboard():
    if not st.session_state.get("admin"):
        st.session_state.page = "admin_login"
        st.rerun()

    st.title("Admin Dashboard")

    con = db()
    total_voters = con.execute("SELECT COUNT(*) FROM voters").fetchone()[0]
    total_votes = con.execute("SELECT COUNT(*) FROM votes").fetchone()[0]
    yet_to_vote = con.execute(
        "SELECT COUNT(*) FROM voters WHERE has_voted=0"
    ).fetchone()[0]

    candidates = con.execute("""
        SELECT c.name,c.party,c.symbol,COUNT(v.id) AS votes
        FROM candidates c
        LEFT JOIN votes v ON c.id=v.candidate_id
        GROUP BY c.id
        ORDER BY c.id
    """).fetchall()
    con.close()

    a, b, c = st.columns(3)
    a.metric("Total Voters", total_voters)
    b.metric("Total Votes", total_votes)
    c.metric("Yet to Vote", yet_to_vote)

    st.subheader("Candidate Results")
    for candidate in candidates:
        col1, col2, col3, col4 = st.columns([1, 3, 3, 1])
        symbol_path = os.path.join(SYMBOL_DIR, candidate["symbol"])
        with col1:
            if os.path.exists(symbol_path):
                st.image(symbol_path, width=50)
        col2.write(candidate["name"])
        col3.write(candidate["party"])
        col4.write(str(candidate["votes"]))

    if st.button("Logout", use_container_width=True):
        st.session_state.clear()
        st.session_state.page = "home"
        st.rerun()

def main():
    initialize_database()

    if "page" not in st.session_state:
        st.session_state.page = "home"

    pages = {
        "home": home,
        "register": register,
        "login": voter_login,
        "vote": vote_page,
        "admin_login": admin_login,
        "admin": admin_dashboard,
    }
    pages[st.session_state.page]()

if __name__ == "__main__":
    main()
