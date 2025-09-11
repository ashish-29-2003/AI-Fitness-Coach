from flask import Flask, render_template, Response, request, send_file, send_from_directory
import cv2
import os

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Global camera object (keeps running for speed)
camera = None

def get_camera(index=0):
    global camera
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(index, cv2.CAP_DSHOW)  # CAP_DSHOW makes startup faster on Windows
    return camera

def gen_frames():
    cap = get_camera()
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

@app.route("/camera")
def camera_page():
    return render_template("camera.html")

@app.route("/camera_feed")
def camera_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

# Route to serve uploads if needed
@app.route('/uploads/<path:filename>')
def uploads(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

if __name__ == "__main__":
    app.run(debug=True)
