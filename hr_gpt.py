import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch, find_peaks
import time
import csv
from datetime import datetime
import threading
from deepface import DeepFace
from collections import deque, Counter

# ----------------------------
# Parameters & Constants
# ----------------------------
fs = 30
window_size = fs * 10
BUFFER_MAX = fs * 15
SMOOTH_ALPHA = 0.2

STRESS_HIGH_THRESH = 25
STRESS_LOW_THRESH = 50

EAR_THRESHOLD = 0.21
DROWSINESS_FRAMES = 45

# ----------------------------
# State Variables
# ----------------------------
green_signal = []
smooth_bpm = []
frame_idx = []
frame_count = 0
last_valid_bpm = None
current_rmssd = 0

stress_level = "CALIBRATING..."
stress_color = (0, 255, 255)

blink_counter = 0
drowsy_alert = False

current_emotion = "NEUTRAL"
emotion_color = (200, 200, 200)
emotion_buffer = deque(maxlen=7)

stop_threads = False
latest_face_for_emotion = None

log_data = []
last_log_time = time.time()

# ----------------------------
# MediaPipe Setup
# ----------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

# ----------------------------
# Landmark Indexes
# ----------------------------
FOREHEAD = [10, 67, 69, 104, 108, 109]
LEFT_CHEEK = [50, 101, 118, 119, 120, 47]
RIGHT_CHEEK = [280, 330, 347, 348, 349, 277]
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# ----------------------------
# Helper Functions
# ----------------------------
def extract_roi_green(frame, lm, idxs, w, h):
    xs = [int(lm[i].x * w) for i in idxs]
    ys = [int(lm[i].y * h) for i in idxs]
    x1, x2 = max(min(xs), 0), min(max(xs), w)
    y1, y2 = max(min(ys), 0), min(max(ys), h)
    if x2 <= x1 or y2 <= y1:
        return None, None
    roi = frame[y1:y2, x1:x2]
    return np.mean(roi[:, :, 1]), (x1, y1, x2, y2)

def calculate_ear(lm, idxs, w, h):
    p2 = np.array([lm[idxs[1]].x * w, lm[idxs[1]].y * h])
    p6 = np.array([lm[idxs[5]].x * w, lm[idxs[5]].y * h])
    p3 = np.array([lm[idxs[2]].x * w, lm[idxs[2]].y * h])
    p5 = np.array([lm[idxs[4]].x * w, lm[idxs[4]].y * h])
    p1 = np.array([lm[idxs[0]].x * w, lm[idxs[0]].y * h])
    p4 = np.array([lm[idxs[3]].x * w, lm[idxs[3]].y * h])
    return (np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / (2.0 * np.linalg.norm(p1 - p4))

def extract_face_roi(frame, lm, w, h):
    xs = [int(p.x * w) for p in lm]
    ys = [int(p.y * h) for p in lm]
    x1, x2 = max(min(xs) - 20, 0), min(max(xs) + 20, w)
    y1, y2 = max(min(ys) - 20, 0), min(max(ys) + 20, h)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]

# ----------------------------
# Emotion Background Thread
# ----------------------------
def emotion_worker():
    global current_emotion, emotion_color, latest_face_for_emotion
    emotion_map = {
        'HAPPY': (0, 255, 0),
        'SAD': (255, 0, 0),
        'ANGRY': (0, 0, 255),
        'FEAR': (128, 0, 128),
        'SURPRISE': (0, 255, 255),
        'DISGUST': (0, 128, 128),
        'NEUTRAL': (200, 200, 200)
    }

    while not stop_threads:
        if latest_face_for_emotion is not None:
            try:
                img = latest_face_for_emotion.copy()
                result = DeepFace.analyze(
                    img_path=img,
                    actions=['emotion'],
                    enforce_detection=False,
                    detector_backend='retinaface',
                    silent=True
                )

                dominant = result[0]['dominant_emotion'].upper()
                emotion_buffer.append(dominant)

                if len(emotion_buffer) >= 5:
                    current_emotion = Counter(emotion_buffer).most_common(1)[0][0]
                    emotion_color = emotion_map.get(current_emotion, (255, 255, 255))

            except:
                pass

            time.sleep(0.4)
        else:
            time.sleep(0.1)

t = threading.Thread(target=emotion_worker, daemon=True)
t.start()

# ----------------------------
# Matplotlib Setup
# ----------------------------
plt.ion()
fig, ax = plt.subplots()
line, = ax.plot([], [], linewidth=2)
ax.set_ylim(40, 140)
ax.grid(True)

# ----------------------------
# Main Loop
# ----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)

    face_detected = False

    if res.multi_face_landmarks:
        face_detected = True
        lm = res.multi_face_landmarks[0].landmark

        face_roi = extract_face_roi(frame, lm, w, h)
        if face_roi is not None:
            latest_face_for_emotion = face_roi

        roi_means = []
        for region in [FOREHEAD, LEFT_CHEEK, RIGHT_CHEEK]:
            val, _ = extract_roi_green(frame, lm, region, w, h)
            if val is not None:
                roi_means.append(val)

        if len(roi_means) >= 2:
            green_signal.append(np.mean(roi_means))
            if len(green_signal) > BUFFER_MAX:
                green_signal.pop(0)

        left_ear = calculate_ear(lm, LEFT_EYE, w, h)
        right_ear = calculate_ear(lm, RIGHT_EYE, w, h)
        avg_ear = (left_ear + right_ear) / 2

        if avg_ear < EAR_THRESHOLD:
            blink_counter += 1
        else:
            blink_counter = 0
            drowsy_alert = False

        if blink_counter >= DROWSINESS_FRAMES:
            drowsy_alert = True

    if face_detected and len(green_signal) >= window_size:
        raw = np.array(green_signal[-window_size:])
        raw -= np.mean(raw)

        b, a = butter(3, [0.7/(fs/2), 3.5/(fs/2)], btype='band')
        filtered = filtfilt(b, a, raw)

        freqs, psd = welch(filtered, fs, nfft=2048)
        valid = (freqs >= 0.7) & (freqs <= 3.5)

        bpm = freqs[valid][np.argmax(psd[valid])] * 60
        last_valid_bpm = bpm if last_valid_bpm is None else SMOOTH_ALPHA*bpm + (1-SMOOTH_ALPHA)*last_valid_bpm

        peaks, _ = find_peaks(filtered, distance=fs*0.4)
        if len(peaks) > 2:
            rr = np.diff(peaks)/fs*1000
            current_rmssd = np.sqrt(np.mean(np.diff(rr)**2))

            if current_rmssd < STRESS_HIGH_THRESH:
                stress_level = "HIGH STRESS"
                stress_color = (0, 0, 255)
            elif current_rmssd < STRESS_LOW_THRESH:
                stress_level = "MEDIUM STRESS"
                stress_color = (0, 255, 255)
            else:
                stress_level = "RELAXED"
                stress_color = (0, 255, 0)

    smooth_bpm.append(last_valid_bpm if last_valid_bpm else 0)
    frame_idx.append(frame_count)
    if len(frame_idx) > 200:
        frame_idx.pop(0)
        smooth_bpm.pop(0)

    line.set_xdata(frame_idx)
    line.set_ydata(smooth_bpm)
    ax.set_xlim(max(0, frame_count-150), frame_count+10)
    plt.pause(0.001)

    cv2.putText(frame, f"BPM: {int(last_valid_bpm) if last_valid_bpm else '--'}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    cv2.putText(frame, f"Stress: {stress_level}", (20,80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, stress_color, 2)
    cv2.putText(frame, f"Mood: {current_emotion}", (20,120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, emotion_color, 2)

    cv2.imshow("Multimodal Health Monitor", frame)
    frame_count += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ----------------------------
# Cleanup
# ----------------------------
stop_threads = True
t.join()
cap.release()
cv2.destroyAllWindows()
plt.ioff()

with open("complete_health_log.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Time","BPM","HRV","Stress","Emotion","Status"])
    writer.writerows(log_data)

plt.show()
