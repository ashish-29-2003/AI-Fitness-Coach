from flask import Flask, request, render_template, Response, redirect, url_for
import cv2
import mediapipe as mp

app = Flask(__name__)

# Global variables
camera_index = 0
cap = None

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# ---- Home ----
@app.route("/")
def index():
    return render_template("index.html")

# ---- Start camera with selected index ----
@app.route("/start_camera", methods=["POST"])
def start_camera():
    global camera_index, cap
    camera_index = int(request.form.get("camera_index", 0))

    # Release any previous capture
    if cap is not None:
        cap.release()

    # Re-initialize with chosen camera
    cap = cv2.VideoCapture(camera_index)

    return redirect(url_for("camera"))

# ---- Camera preview page ----
@app.route("/camera")
def camera():
    return render_template("camera.html")

# ---- Stream feed ----
def generate_frames():
    global cap
    if cap is None:
        cap = cv2.VideoCapture(camera_index)

    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5) as pose:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # Pose detection
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Encode frame
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route("/camera_feed")
def camera_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

# ---- Cleanup ----
@app.route("/stop_camera")
def stop_camera():
    global cap
    if cap is not None:
        cap.release()
        cap = None
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)
