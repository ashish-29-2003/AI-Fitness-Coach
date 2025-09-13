import cv2
import mediapipe as mp
import numpy as np

# State dictionary to hold counters and stages between frames
exercise_state = {
    'pushup_counter': 0, 'squat_counter': 0, 'jumping_jack_counter': 0,
    'pushup_stage': None, 'squat_stage': None, 'jumping_jack_stage': None
}

# Initialize MediaPipe components
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils # The drawing utility
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def calculate_angle(a, b, c):
    """Calculates the angle between three points."""
    a = np.array(a); b = np.array(b); c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return angle if angle <= 180.0 else 360 - angle

def process_live_frame(frame):
    """
    Processes a single frame for live exercise counting and adds visual feedback.
    Returns the annotated frame and the current rep counts.
    """
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    try:
        landmarks = results.pose_landmarks.landmark
        
        # --- Push-up Logic ---
        shoulder_l = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        elbow_l = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        wrist_l = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        angle_elbow_l = calculate_angle(shoulder_l, elbow_l, wrist_l)
        if angle_elbow_l > 160: exercise_state['pushup_stage'] = "down"
        if angle_elbow_l < 40 and exercise_state['pushup_stage'] == 'down':
            exercise_state['pushup_stage'] = "up"; exercise_state['pushup_counter'] += 1

        # --- Squat Logic ---
        hip_r = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        knee_r = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        ankle_r = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
        angle_knee_r = calculate_angle(hip_r, knee_r, ankle_r)
        if angle_knee_r > 160: exercise_state['squat_stage'] = "up"
        if angle_knee_r < 90 and exercise_state['squat_stage'] == 'up':
            exercise_state['squat_stage'] = 'down'; exercise_state['squat_counter'] += 1
            
        # --- Jumping Jack Logic ---
        if landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y < landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y:
            exercise_state['jumping_jack_stage'] = "up"
        if landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y > landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y and exercise_state['jumping_jack_stage'] == 'up':
            exercise_state['jumping_jack_stage'] = "down"; exercise_state['jumping_jack_counter'] += 1

    except Exception:
        pass

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
        )

    rep_counts = {
        "pushups": exercise_state['pushup_counter'],
        "squats": exercise_state['squat_counter'],
        "jumping_jacks": exercise_state['jumping_jack_counter']
    }
    return image, rep_counts

