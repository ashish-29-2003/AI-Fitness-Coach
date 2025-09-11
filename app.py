from flask import Flask, request, render_template, send_file, send_from_directory, Response, url_for
import os
import cv2
import mediapipe as mp
import matplotlib
matplotlib.use('Agg')  # no GUI
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# ---------------- HOME ----------------
@app.route("/")
def index():
    return render_template("index.html")

# ---------------- UPLOAD VIDEO ----------------
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
    detected_frames, snapshot_saved = 0, False
    snapshot_path = os.path.join(app.config["UPLOAD_FOLDER"], "snapshot.png")
    accuracy_over_time = []

    # Exercise counters
    pushups = squats = jumping_jacks = 0

    with mp_pose.Pose(min_detection_confidence=0.5) as pose:
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

                # Demo counters
                if frame_idx % 100 == 0: pushups += 1
                if frame_idx % 120 == 0: squats += 1
                if frame_idx % 150 == 0: jumping_jacks += 1

            if not snapshot_saved and frame_idx == frame_count // 2:
                cv2.imwrite(snapshot_path, frame)
                snapshot_saved = True

            accuracy_over_time.append(round((detected_frames / frame_idx) * 100, 2))

    cap.release()
    accuracy = round((detected_frames / frame_count) * 100, 2)

    # Accuracy chart
    chart_path = os.path.join(app.config["UPLOAD_FOLDER"], "accuracy_chart.png")
    plt.figure()
    plt.plot(range(1, len(accuracy_over_time) + 1), accuracy_over_time, color='blue')
    plt.xlabel("Frame")
    plt.ylabel("Accuracy (%)")
    plt.title("Detection Accuracy Over Time")
    plt.savefig(chart_path)
    plt.close()

    # Pie chart
    pie_path = os.path.join(app.config["UPLOAD_FOLDER"], "exercise_pie.png")
    plt.figure()
    plt.pie([pushups, squats, jumping_jacks], labels=["Push-ups", "Squats", "Jumping Jacks"],
            autopct="%1.1f%%", startangle=140)
    plt.title("Exercise Distribution")
    plt.savefig(pie_path)
    plt.close()

    global analysis_result
    analysis_result = {
        "frame_count": frame_count,
        "fps": round(fps, 2),
        "detected_frames": detected_frames,
        "accuracy": accuracy,
        "snapshot_path": f"/uploads/{os.path.basename(snapshot_path)}",
        "chart_path": f"/uploads/{os.path.basename(chart_path)}",
        "pie_chart": f"/uploads/{os.path.basename(pie_path)}",
        "pushups": pushups,
        "squats": squats,
        "jumping_jacks": jumping_jacks
    }

    return render_template("result.html", **analysis_result)

# ---------------- CAMERA MODE ----------------
def generate_camera_feed(cam_index):
    cap = cv2.VideoCapture(cam_index)
    with mp_pose.Pose(min_detection_confidence=0.5) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route("/start_camera")
def start_camera():
    cam_index = int(request.args.get("camera_index", 0))
    return render_template("camera.html", stream_url=url_for('camera_feed', cam_index=cam_index), cam_index=cam_index)

@app.route("/camera_feed")
def camera_feed():
    cam_index = int(request.args.get("cam_index", 0))
    return Response(generate_camera_feed(cam_index), mimetype="multipart/x-mixed-replace; boundary=frame")

# ---------------- PDF ----------------
@app.route("/download_pdf")
def download_pdf():
    global analysis_result
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], "analysis_report.pdf")
    c = canvas.Canvas(filepath, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 20)
    c.drawCentredString(width/2, height-80, "Workout Analysis Report")
    c.line(50, height-100, width-50, height-100)

    y = height - 140
    c.setFont("Helvetica", 12)
    for key, label in [
        ("frame_count", "Total Frames"),
        ("fps", "FPS"),
        ("detected_frames", "Detected Frames"),
        ("accuracy", "Accuracy (%)"),
        ("pushups", "Push-ups"),
        ("squats", "Squats"),
        ("jumping_jacks", "Jumping Jacks")
    ]:
        c.drawString(80, y, f"{label}: {analysis_result[key]}")
        y -= 20

    row_y = height - 400
    image_w, image_h = 160, 160
    x_positions = [70, 230, 390]
    for path, pos in zip(
        [analysis_result["snapshot_path"], analysis_result["chart_path"], analysis_result["pie_chart"]],
        x_positions
    ):
        full_path = os.path.join(app.config["UPLOAD_FOLDER"], os.path.basename(path))
        if os.path.exists(full_path):
            c.drawImage(ImageReader(full_path), pos, row_y, width=image_w, height=image_h)

    c.save()
    return send_file(filepath, as_attachment=True)

# ---------------- Serve Uploads ----------------
@app.route('/uploads/<path:filename>')
def uploads(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

if __name__ == "__main__":
    app.run(debug=True)
