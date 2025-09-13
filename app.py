from flask import Flask, render_template, request, redirect, url_for, flash, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_socketio import SocketIO
from datetime import datetime
import cv2
import os
import numpy as np
import base64
from werkzeug.utils import secure_filename
# Correctly import the two different processing functions
from pose_estimation import process_live_frame
from pose_estimation_video import analyze_video 

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SECRET_KEY'] = 'a_secret_key_for_flash_messages'
app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{os.path.join(app.instance_path, 'fitness_history.db')}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

socketio = SocketIO(app)

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.instance_path, exist_ok=True)

db = SQLAlchemy(app)

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

# --- UPDATED WebSocket Event Handler ---
@socketio.on('frame')
def handle_frame(data_image):
    # Decode the base64 image data from the browser
    sbuf = base64.b64decode(data_image.split(',')[1])
    nparr = np.frombuffer(sbuf, dtype=np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Process the frame to get the annotated image and rep counts
    annotated_frame, rep_counts = process_live_frame(frame)

    # Encode the annotated frame back to JPEG format
    _, buffer = cv2.imencode('.jpg', annotated_frame)
    # Encode the buffer to a base64 string
    encoded_image = base64.b64encode(buffer).decode('utf-8')

    # Send both the annotated image and the rep counts back to the client
    socketio.emit('response', {'image': encoded_image, 'counts': rep_counts})

# --- Standard HTTP Routes (No changes below this line) ---
@app.route("/")
def index():
    return render_template("index.html")

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        flash('No file part'); return redirect(request.url)
    file = request.files['video']
    if file.filename == '':
        flash('No selected file'); return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        cap = cv2.VideoCapture(filepath)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        analysis_results = analyze_video(filepath)
        processed_frames = analysis_results['frame_count'] // 5
        accuracy = (analysis_results['detected_frames'] / processed_frames) * 100 if processed_frames > 0 else 0
        
        new_workout = Workout(
            frames=analysis_results['frame_count'], fps=round(fps, 2) if fps else 0,
            detected_frames=analysis_results['detected_frames'], accuracy=round(accuracy, 2),
            pushups=analysis_results['pushups'], squats=analysis_results['squats'],
            jumping_jacks=analysis_results['jumping_jacks'], video_path=filepath 
        )
        db.session.add(new_workout); db.session.commit()

        context = {
            'workout_id': new_workout.id, 'frame_count': new_workout.frames, 'fps': new_workout.fps,
            'detected_frames': new_workout.detected_frames, 'accuracy': new_workout.accuracy,
            'pushups': new_workout.pushups, 'squats': new_workout.squats, 'jumping_jacks': new_workout.jumping_jacks,
            'snapshot_path': url_for('static', filename='accuracy_plot.png'),
            'chart_path': url_for('static', filename='accuracy_plot.png'),
            'pie_chart': url_for('static', filename='accuracy_plot.png')
        }
        return render_template('result.html', **context)
    return redirect(url_for('index'))

def generate_pdf(workout):
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Spacer
    from reportlab.lib import colors
    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], f"workout_report_{workout.id}.pdf")
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    elements = [Table([["Workout Analysis Report"]], style=[('ALIGN', (0,0), (-1,-1), 'CENTER'), ('FONTSIZE', (0,0), (-1,-1), 18)])]
    elements.append(Spacer(1, 24))
    data = [["Metric", "Value"], ["Date", workout.date.strftime("%Y-%m-%d %H:%M")], ["Total Frames", str(workout.frames)],
            ["Accuracy", f"{workout.accuracy:.2f}%"], ["Push-ups", str(workout.pushups)], ["Squats", str(workout.squats)], ["Jumping Jacks", str(workout.jumping_jacks)]]
    table = Table(data, colWidths=[200, 200])
    table.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.grey), ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke), ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'), ('BOTTOMPADDING', (0,0), (-1,0), 12), ('BACKGROUND', (0,1), (-1,-1), colors.beige), ('GRID', (0,0), (-1,-1), 1, colors.black)]))
    elements.append(table)
    doc.build(elements)
    workout.pdf_path = pdf_path
    db.session.commit()
    return pdf_path

@app.route("/download_pdf/<int:workout_id>")
def download_pdf(workout_id):
    workout = Workout.query.get_or_404(workout_id)
    if not workout.pdf_path or not os.path.exists(workout.pdf_path):
        generate_pdf(workout)
    return send_file(workout.pdf_path, as_attachment=True)

@app.route("/history")
def history():
    workouts = Workout.query.order_by(Workout.date.desc()).all()
    return render_template("history.html", workouts=workouts)

@app.route("/camera")
def camera_page():
    return render_template("camera.html")

if __name__ == "__main__":
    socketio.run(app, debug=True, use_reloader=False)

