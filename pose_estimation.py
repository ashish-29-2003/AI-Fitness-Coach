import cv2
import mediapipe as mp
import numpy as np

# --- State Management ---
exercise_state = {
    'pushup_counter': 0, 'squat_counter': 0, 'jumping_jack_counter': 0,
    'pushup_stage': None, 'squat_stage': None, 'jumping_jack_stage': None
}

# Initialize MediaPipe Pose and Drawing utilities
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def reset_state():
    """Resets all counters and stages."""
    global exercise_state
    for key in exercise_state:
        if 'counter' in key:
            exercise_state[key] = 0
        else:
            exercise_state[key] = None
    print("Counters have been reset.")

def calculate_angle(a, b, c):
    """Calculates the angle between three points."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return angle if angle <= 180.0 else 360 - angle

def process_live_frame(frame, exercise_type):
    """Processes a single frame with improved, more robust exercise logic."""
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    try:
        landmarks = results.pose_landmarks.landmark
        
        if exercise_type == 'pushups':
            shoulder_l = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow_l = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist_l = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            shoulder_r = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbow_r = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            wrist_r = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            # Calculate angles for both elbows
            angle_elbow_l = calculate_angle(shoulder_l, elbow_l, wrist_l)
            angle_elbow_r = calculate_angle(shoulder_r, elbow_r, wrist_r)
            
            # --- PUSH-UP LOGIC FIX ---
            # A repetition is counted when the user goes from "down" to "up".
            # Stage is set to "down" when arms are bent.
            if angle_elbow_l < 50 and angle_elbow_r < 50:
                exercise_state['pushup_stage'] = "down"
            
            # Stage is set to "up" and rep is counted when arms are extended *after* being down.
            if (angle_elbow_l > 160 and angle_elbow_r > 160) and exercise_state['pushup_stage'] == 'down':
                exercise_state['pushup_stage'] = "up"
                exercise_state['pushup_counter'] += 1

        elif exercise_type == 'squats':
            hip_r = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            knee_r = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            ankle_r = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            hip_l = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee_l = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

            # Calculate angles for both knees
            angle_knee_r = calculate_angle(hip_r, knee_r, ankle_r)
            angle_knee_l = calculate_angle(hip_l, knee_l, ankle_r)

            # --- SQUAT LOGIC FIX ---
            # A repetition is counted when the user goes from "down" to "up".
            # Stage is set to "down" when hips are below knees (a deep squat).
            if (hip_r[1] > knee_r[1] and hip_l[1] > knee_l[1]):
                exercise_state['squat_stage'] = "down"
            
            # Stage is set to "up" and rep is counted when standing straight *after* being down.
            if (angle_knee_r > 160 and angle_knee_l > 160) and exercise_state['squat_stage'] == 'down':
                exercise_state['squat_stage'] = 'up'
                exercise_state['squat_counter'] += 1
            
        elif exercise_type == 'jumping_jacks':
            shoulder_l = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            wrist_l = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            ankle_l = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            ankle_r = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            
            is_down = wrist_l[1] > shoulder_l[1] and abs(ankle_l[0] - ankle_r[0]) < 0.15
            if is_down:
                 exercise_state['jumping_jack_stage'] = "down"

            is_up = wrist_l[1] < shoulder_l[1] and abs(ankle_l[0] - ankle_r[0]) > 0.2
            if is_up and exercise_state['jumping_jack_stage'] == 'down':
                exercise_state['jumping_jack_stage'] = "up"
                exercise_state['jumping_jack_counter'] += 1

    except Exception:
        pass

    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
    
    rep_counts = {
        "pushups": exercise_state['pushup_counter'],
        "squats": exercise_state['squat_counter'],
        "jumping_jacks": exercise_state['jumping_jack_counter']
    }

    return image, rep_counts

