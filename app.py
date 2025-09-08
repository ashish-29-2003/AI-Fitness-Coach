from flask import Flask, request, render_template, send_file, send_from_directory, redirect, url_for
import os
import cv2
import mediapipe as mp
import csv
import matplotlib
matplotlib.use('Agg')  # For servers without display
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

# Flask app + Database setup
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///fitness_history.db"
db = SQLAlchemy(app)

# Database model
class Workout(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.DateTime, default=datetime.utcnow)
    frames = db.Column(db.Integer)
    fps = db.Column(db.Float)
    detected_frames = db.Column(db.Integer)
    accuracy = db.Column(db.Float)
    pushups = db.Column(db.Integer)
    squats = db.Column(db.Integer)
    jumping_jacks = db.Column(db.Integer)
    video_path = db.Column(db.String(200))
    snapshot_path = db.Column(db.String(200))
    pdf_path = db.Column(db.String(200))
    csv_path = db.Column(db.String(200))

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "video" not in request.files:
        return "No file uploaded"

    file = request.files["video"]
    if file.filename == "":
        return "No selected file"

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        return "Could not process video"

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    processed_video_path = os.path.join(app.config["UPLOAD_FOLDER"], "processed_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(processed_video_path, fourcc, fps, (width, height))

    detected_frames = 0
    accuracy_over_time = []
    snapshot_saved = False
    snapshot_path = os.path.join(app.config["UPLOAD_FOLDER"], "snapshot.png")

    pushups, squats, jumping_jacks = 0, 0, 0

    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5) as pose:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if results.pose_landmarks:
                detected_frames += 1
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # Demo counters (replace with real detection later)
                if frame_idx % 100 == 0: pushups += 1
                if frame_idx % 120 == 0: squats += 1
                if frame_idx % 150 == 0: jumping_jacks += 1

            if not snapshot_saved and frame_idx == frame_count // 2:
                cv2.imwrite(snapshot_path, frame)
                snapshot_saved = True

            out.write(frame)
            accuracy_over_time.append(round((detected_frames / frame_idx) * 100, 2))

    cap.release()
    out.release()

    accuracy = round((detected_frames / frame_count) * 100, 2)

    # Accuracy chart
    chart_path = os.path.join(app.config["UPLOAD_FOLDER"], "accuracy_chart.png")
    plt.figure()
    plt.plot(range(1, len(accuracy_over_time) + 1), accuracy_over_time, color='blue', label="Accuracy")
    plt.xlabel("Frame")
    plt.ylabel("Accuracy (%)")
    plt.title("Detection Accuracy Over Time")
    plt.legend()
    plt.savefig(chart_path)
    plt.close()

    # Pie chart
    labels = ["Push-ups", "Squats", "Jumping Jacks"]
    values = [pushups, squats, jumping_jacks]
    pie_path = os.path.join(app.config["UPLOAD_FOLDER"], "exercise_pie.png")
    plt.figure()
    plt.pie(values, labels=labels, autopct="%1.1f%%", startangle=140)
    plt.title("Exercise Distribution")
    plt.savefig(pie_path)
    plt.close()

    # Save results globally
    global analysis_result
    analysis_result = {
        "frame_count": frame_count,
        "fps": round(fps, 2),
        "detected_frames": detected_frames,
        "accuracy": accuracy,
        "accuracy_over_time": accuracy_over_time,
        "processed_video_path": processed_video_path,
        "snapshot_path": snapshot_path,
        "pushups": pushups,
        "squats": squats,
        "jumping_jacks": jumping_jacks,
        "chart_path": "accuracy_chart.png",
        "pie_chart": "exercise_pie.png"
    }

    # Save to DB
    workout = Workout(
        frames=frame_count, fps=round(fps, 2),
        detected_frames=detected_frames, accuracy=accuracy,
        pushups=pushups, squats=squats, jumping_jacks=jumping_jacks,
        video_path=processed_video_path, snapshot_path=snapshot_path,
        pdf_path=os.path.join(app.config["UPLOAD_FOLDER"], "analysis_report.pdf"),
        csv_path=os.path.join(app.config["UPLOAD_FOLDER"], "analysis_result.csv")
    )
    db.session.add(workout)
    db.session.commit()

    return render_template("result.html", **analysis_result)

@app.route("/history")
def history():
    workouts = Workout.query.order_by(Workout.date.desc()).all()
    return render_template("history.html", workouts=workouts)

@app.route("/download_csv")
def download_csv():
    global analysis_result
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], "analysis_result.csv")
    with open(filepath, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Total Frames", "FPS", "Detected Frames", "Accuracy (%)", "Push-ups", "Squats", "Jumping Jacks"])
        writer.writerow([
            analysis_result["frame_count"],
            analysis_result["fps"],
            analysis_result["detected_frames"],
            analysis_result["accuracy"],
            analysis_result["pushups"],
            analysis_result["squats"],
            analysis_result["jumping_jacks"]
        ])
    return send_file(filepath, as_attachment=True)

@app.route("/download_pdf")
def download_pdf():
    global analysis_result
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], "analysis_report.pdf")
    c = canvas.Canvas(filepath, pagesize=letter)
    width, height = letter
    c.setFont("Helvetica-Bold", 20)
    c.drawCentredString(width / 2, height - 80, "Workout Analysis Report")
    c.setFont("Helvetica", 12)
    c.drawCentredString(width / 2, height - 110, "Generated by AI Fitness Coach")
    c.line(50, height - 120, width - 50, height - 120)
    c.setFont("Helvetica", 14)
    y = height - 160
    c.drawString(80, y, f"Total Frames: {analysis_result['frame_count']}")
    y -= 30; c.drawString(80, y, f"FPS: {analysis_result['fps']}")
    y -= 30; c.drawString(80, y, f"Person Detected Frames: {analysis_result['detected_frames']}")
    y -= 30; c.drawString(80, y, f"Accuracy: {analysis_result['accuracy']} %")
    y -= 30; c.drawString(80, y, f"Push-ups: {analysis_result['pushups']}")
    y -= 30; c.drawString(80, y, f"Squats: {analysis_result['squats']}")
    y -= 30; c.drawString(80, y, f"Jumping Jacks: {analysis_result['jumping_jacks']}")
    if os.path.exists(analysis_result["snapshot_path"]):
        y -= 250
        c.drawImage(ImageReader(analysis_result["snapshot_path"]), 80, y, width=450, height=250)
    c.setFont("Helvetica-Oblique", 10)
    c.drawCentredString(width / 2, 50, "This report was auto-generated.")
    c.save()
    return send_file(filepath, as_attachment=True)

@app.route("/download_video")
def download_video():
    global analysis_result
    return send_file(analysis_result["processed_video_path"], as_attachment=True)

@app.route('/uploads/<path:filename>')
def uploads(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
