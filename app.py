from flask import Flask, request, render_template
import os
import cv2
import mediapipe as mp

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

mp_pose = mp.solutions.pose

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
    detected_frames = 0

    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            if results.pose_landmarks:
                detected_frames += 1

    cap.release()
    accuracy = round((detected_frames / frame_count) * 100, 2)

    return render_template(
        "result.html",
        frame_count=frame_count,
        fps=round(fps, 2),
        detected_frames=detected_frames,
        accuracy=accuracy
    )

if __name__ == "__main__":
    app.run(debug=True)
