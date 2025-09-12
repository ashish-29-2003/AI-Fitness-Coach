from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import cv2
import os
from werkzeug.utils import secure_filename
from pose_estimation import analyze_video

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SECRET_KEY'] = 'a_secret_key_for_flash_messages'
app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{os.path.join(app.instance_path, 'fitness_history.db')}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Ensure the upload and instance folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.instance_path, exist_ok=True)

db = SQLAlchemy(app)

# Database Model
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
    video_path = db.Column(db.String(200), nullable=True)
    snapshot_path = db.Column(db.String(200), nullable=True)
    pdf_path = db.Column(db.String(200), nullable=True)
    csv_path = db.Column(db.String(200), nullable=True)

with app.app_context():
    db.create_all()

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['video']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        cap = cv2.VideoCapture(filepath)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        analysis_results = analyze_video(filepath)

        if analysis_results['frame_count'] > 0:
            accuracy = (analysis_results['detected_frames'] / analysis_results['frame_count']) * 100
        else:
            accuracy = 0
        
        new_workout = Workout(
            frames=analysis_results['frame_count'],
            fps=round(fps, 2) if fps else 0,
            detected_frames=analysis_results['detected_frames'],
            accuracy=round(accuracy, 2),
            pushups=analysis_results['pushups'],
            squats=analysis_results['squats'],
            jumping_jacks=analysis_results['jumping_jacks'],
            video_path=filepath 
        )
        db.session.add(new_workout)
        db.session.commit()

        context = {
            'frame_count': new_workout.frames,
            'fps': new_workout.fps,
            'detected_frames': new_workout.detected_frames,
            'accuracy': new_workout.accuracy,
            'pushups': new_workout.pushups,
            'squats': new_workout.squats,
            'jumping_jacks': new_workout.jumping_jacks,
            'snapshot_path': url_for('static', filename='accuracy_plot.png'),
            'chart_path': url_for('static', filename='accuracy_plot.png'),
            'pie_chart': url_for('static', filename='accuracy_plot.png')
        }
        return render_template('result.html', **context)

    return redirect(url_for('index'))

@app.route("/history")
def history():
    workouts = Workout.query.order_by(Workout.date.desc()).all()
    return render_template("history.html", workouts=workouts)

@app.route("/camera")
def camera_page():
    # This route now simply serves the HTML page.
    # The JavaScript on that page handles the camera.
    return render_template("camera.html")

if __name__ == "__main__":
    app.run(debug=True)

