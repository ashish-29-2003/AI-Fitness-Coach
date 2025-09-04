import os
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

# Make sure uploads folder exists
UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv"}  # Allowed video formats

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return "No file part", 400

    file = request.files["video"]
    if file.filename == "":
        return "No selected file", 400

    if file and allowed_file(file.filename):
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)
        return f"✅ Video uploaded successfully! Saved at {filepath}"

    return "File type not allowed", 400

if __name__ == "__main__":
    app.run(debug=True)
