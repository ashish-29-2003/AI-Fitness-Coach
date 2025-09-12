import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    """Calculates the angle between three points."""
    a = np.array(a)  # First point
    b = np.array(b)  # Mid point
    c = np.array(c)  # End point

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

def analyze_video(video_path):
    """
    Analyzes a video file to count push-ups, squats, and jumping jacks.
    """
    cap = cv2.VideoCapture(video_path)

    # --- OPTIMIZATION ---
    # We will process only 1 in every 5 frames to make analysis faster.
    FRAME_SKIP = 5

    # Exercise counters and stages
    pushup_counter = 0
    squat_counter = 0
    jumping_jack_counter = 0
    pushup_stage = None
    squat_stage = None
    jumping_jack_stage = None

    frame_count = 0
    detected_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # --- OPTIMIZATION ---
        # If the current frame number is not a multiple of FRAME_SKIP, skip it.
        if frame_count % FRAME_SKIP != 0:
            continue

        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks if pose is detected
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            detected_frames += 1

            # --- Push-up Logic ---
            shoulder_l = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow_l = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist_l = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            angle_elbow_l = calculate_angle(shoulder_l, elbow_l, wrist_l)
            
            if angle_elbow_l > 160:
                pushup_stage = "down"
            if angle_elbow_l < 40 and pushup_stage == 'down':
                pushup_stage = "up"
                pushup_counter += 1

            # --- Squat Logic ---
            hip_r = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            knee_r = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            ankle_r = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            angle_knee_r = calculate_angle(hip_r, knee_r, ankle_r)

            if angle_knee_r > 160:
                squat_stage = "up"
            if angle_knee_r < 90 and squat_stage == 'up':
                squat_stage = 'down'
                squat_counter += 1

            # --- Jumping Jack Logic ---
            if landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y < landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y and \
               landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y < landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y:
                jumping_jack_stage = "up"
            
            if landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y > landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y and \
               landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y > landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y and \
               jumping_jack_stage == 'up':
                jumping_jack_stage = "down"
                jumping_jack_counter += 1
    
    cap.release()

    return {
        "frame_count": frame_count,
        "detected_frames": detected_frames,
        "pushups": pushup_counter,
        "squats": squat_counter,
        "jumping_jacks": jumping_jack_counter
    }

