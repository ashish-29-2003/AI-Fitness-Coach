from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import os

app = Flask(__name__)

# Store camera state
camera = None
camera_index = 0  # default

def get_camera(index=0):
    global camera, camera_index
    if camera is None or camera_index != index or not camera.isOpened():
        if camera is not None:
            camera.release()
        camera = cv2.VideoCapture(index, cv2.CAP_DSHOW)  # faster init on Windows
        camera_index = index
    return camera

def gen_frames():
    cap = get_camera(camera_index)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/camera", methods=["GET", "POST"])
def camera_page():
    if request.method == "POST":
        selected_index = int(request.form.get("camera_index", 0))
        get_camera(selected_index)  # switch camera
        return redirect(url_for("camera_page"))
    return render_template("camera.html", current_index=camera_index)

@app.route("/camera_feed")
def camera_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(debug=True)
