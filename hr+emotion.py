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

# ----------------------------
# Parameters & Constants
# ----------------------------
fs = 30                         # Webcam FPS
window_size = fs * 10           # 10-second window for Heart Rate
BUFFER_MAX = fs * 15            # Buffer limit
SMOOTH_ALPHA = 0.2              # Smoothing factor

# Stress Thresholds (RMSSD in ms)
STRESS_HIGH_THRESH = 25         # < 25ms = High Stress
STRESS_LOW_THRESH = 50          # > 50ms = Relaxed

# Drowsiness Thresholds
EAR_THRESHOLD = 0.21            # < 0.21 means eyes are closed
DROWSINESS_FRAMES = 45          # 1.5 seconds

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

# Drowsiness Variables
blink_counter = 0
drowsy_alert = False

# Emotion Variables
current_emotion = "Neutral"     # Default
emotion_color = (255, 255, 255)
stop_threads = False            # Flag to kill background thread safely

# Data Logging
log_data = []                   # Stores data to save to CSV
last_log_time = time.time()

# ----------------------------
# MediaPipe Setup
# ----------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

# ROI Landmarks
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
    if x2 <= x1 or y2 <= y1: return None, None
    roi = frame[y1:y2, x1:x2]
    return np.mean(roi[:, :, 1]), (x1, y1, x2, y2)

def calculate_ear(lm, idxs, w, h):
    p2 = np.array([lm[idxs[1]].x * w, lm[idxs[1]].y * h])
    p6 = np.array([lm[idxs[5]].x * w, lm[idxs[5]].y * h])
    p3 = np.array([lm[idxs[2]].x * w, lm[idxs[2]].y * h])
    p5 = np.array([lm[idxs[4]].x * w, lm[idxs[4]].y * h])
    p1 = np.array([lm[idxs[0]].x * w, lm[idxs[0]].y * h])
    p4 = np.array([lm[idxs[3]].x * w, lm[idxs[3]].y * h])
    
    v1 = np.linalg.norm(p2 - p6)
    v2 = np.linalg.norm(p3 - p5)
    horiz = np.linalg.norm(p1 - p4)
    return (v1 + v2) / (2.0 * horiz)

# ----------------------------
# Background Thread for Emotion
# ----------------------------
# Variable to hold the CROPPED face image
latest_face_crop = None

def emotion_worker():
    global current_emotion, emotion_color, latest_face_crop, stop_threads
    
    while not stop_threads:
        if latest_face_crop is not None:
            try:
                # Use a local copy of the cropped face
                img_copy = latest_face_crop.copy()
                
                # DeepFace Analysis
                # CRITICAL CHANGE: We use detector_backend='skip' because 
                # we are now feeding it a pre-cropped face from MediaPipe!
                objs = DeepFace.analyze(
                    img_path=img_copy, 
                    actions=['emotion'], 
                    enforce_detection=False,
                    detector_backend='skip', # Skip detection, trust our crop
                    silent=True
                )
                
                if len(objs) > 0:
                    dominant = objs[0]['dominant_emotion']
                    current_emotion = dominant.upper()
                    
                    if current_emotion == 'HAPPY': emotion_color = (0, 255, 0)
                    elif current_emotion == 'SAD': emotion_color = (255, 0, 0)
                    elif current_emotion == 'ANGRY': emotion_color = (0, 0, 255)
                    elif current_emotion == 'NEUTRAL': emotion_color = (200, 200, 200)
                    elif current_emotion == 'FEAR': emotion_color = (128, 0, 128)
                    elif current_emotion == 'SURPRISE': emotion_color = (255, 165, 0)
                    else: emotion_color = (255, 255, 0)
                    
            except Exception as e:
                pass 
            
            # Sleep slightly longer to avoid CPU overload
            time.sleep(0.5)
        else:
            time.sleep(0.1)

# Start thread
t = threading.Thread(target=emotion_worker)
t.daemon = True
t.start()

# ----------------------------
# Matplotlib Setup
# ----------------------------
plt.ion()
fig, ax = plt.subplots()
line, = ax.plot([], [], linewidth=2)
ax.set_xlabel("Time (Frames)")
ax.set_ylabel("Heart Rate (BPM)")
ax.set_title("Psychophysiological Monitor")
ax.set_ylim(40, 140)
ax.grid(True)

print("System Started... Press 'q' to exit.")

# ----------------------------
# Main Loop
# ----------------------------
while True:
    ret, frame = cap.read()
    if not ret: break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)
    face_detected = False
    
    if res.multi_face_landmarks:
        face_detected = True
        lm = res.multi_face_landmarks[0].landmark
        
        # --- NEW: Extract Face Bounding Box for Emotion AI ---
        # Get coords of all landmarks to find the bounding box of the face
        x_coords = [l.x for l in lm]
        y_coords = [l.y for l in lm]
        
        # Add some padding (margin) so we don't crop too tight
        x1 = int(min(x_coords) * w) - 20
        y1 = int(min(y_coords) * h) - 20
        x2 = int(max(x_coords) * w) + 20
        y2 = int(max(y_coords) * h) + 20
        
        # Ensure valid coordinates
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        
        # Crop the face and send to the worker thread
        if x2 > x1 and y2 > y1:
            latest_face_crop = frame[y1:y2, x1:x2]
        # -----------------------------------------------------

        # 1. Heart Rate Extraction
        roi_means = []
        for region, color in zip([FOREHEAD, LEFT_CHEEK, RIGHT_CHEEK],
                                 [(0,255,0), (255,0,0), (0,0,255)]):
            val, box = extract_roi_green(frame, lm, region, w, h)
            if val is not None:
                roi_means.append(val)
                cv2.rectangle(frame, box[:2], box[2:], color, 1)

        if len(roi_means) >= 2:
            green_signal.append(np.mean(roi_means))
            if len(green_signal) > BUFFER_MAX: green_signal.pop(0)

        # 2. Drowsiness (EAR)
        left_ear = calculate_ear(lm, LEFT_EYE, w, h)
        right_ear = calculate_ear(lm, RIGHT_EYE, w, h)
        avg_ear = (left_ear + right_ear) / 2.0
        
        if avg_ear < EAR_THRESHOLD: blink_counter += 1
        else:
            blink_counter = 0
            drowsy_alert = False
            
        if blink_counter >= DROWSINESS_FRAMES:
            drowsy_alert = True
            cv2.putText(frame, "DROWSINESS ALERT!", (200, 300),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

    else:
        green_signal = []
        last_valid_bpm = None
        stress_level = "NO FACE"
        blink_counter = 0

    # 3. Signal Processing
    plot_color = 'gray'
    if face_detected and len(green_signal) >= window_size:
        raw_data = np.array(green_signal[-window_size:])
        normalized = raw_data - np.mean(raw_data)
        
        nyquist = fs / 2
        b, a = butter(3, [0.7/nyquist, 3.5/nyquist], btype='band')
        filtered = filtfilt(b, a, normalized)

        freqs, psd = welch(filtered, fs, nperseg=len(filtered), nfft=2048)
        valid_idxs = np.where((freqs >= 0.7) & (freqs <= 3.5))
        valid_psd = psd[valid_idxs]
        
        if len(valid_psd) > 0:
            peak_idx = np.argmax(valid_psd)
            raw_bpm = freqs[valid_idxs][peak_idx] * 60
            if last_valid_bpm is None: last_valid_bpm = raw_bpm
            else: last_valid_bpm = (SMOOTH_ALPHA * raw_bpm) + ((1 - SMOOTH_ALPHA) * last_valid_bpm)

        peaks, _ = find_peaks(filtered, distance=fs*0.4)
        if len(peaks) > 1:
            diff_rr = np.diff(np.diff(peaks) / fs * 1000)
            if len(diff_rr) > 0:
                current_rmssd = np.sqrt(np.mean(diff_rr ** 2))
            
            if current_rmssd < STRESS_HIGH_THRESH:
                stress_level = "HIGH STRESS"
                plot_color = 'red'
                stress_color = (0, 0, 255)
            elif current_rmssd < STRESS_LOW_THRESH:
                stress_level = "MEDIUM STRESS"
                plot_color = 'orange'
                stress_color = (0, 255, 255)
            else:
                stress_level = "RELAXED"
                plot_color = 'green'
                stress_color = (0, 255, 0)

    # 4. Graph Update
    smooth_bpm.append(last_valid_bpm if last_valid_bpm else 0)
    frame_idx.append(frame_count)
    if len(frame_idx) > 200:
        frame_idx.pop(0)
        smooth_bpm.pop(0)

    line.set_color(plot_color)
    line.set_xdata(frame_idx)
    line.set_ydata(smooth_bpm)
    ax.set_xlim(max(0, frame_count - 150), frame_count + 10)
    plt.pause(0.001)

    # 5. UI Overlay
    cv2.rectangle(frame, (0, 0), (350, 220), (0,0,0), -1)
    
    bpm_text = int(last_valid_bpm) if last_valid_bpm else "--"
    cv2.putText(frame, f"BPM: {bpm_text}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"HRV: {int(current_rmssd)} ms", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    cv2.putText(frame, f"Stress: {stress_level}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, stress_color, 2)
    
    # Show Emotion
    cv2.putText(frame, f"Mood: {current_emotion}", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, emotion_color, 2)

    if face_detected:
        cv2.putText(frame, f"Eye Ratio: {avg_ear:.2f}", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

    # 6. Data Logging (Every 1s)
    if time.time() - last_log_time > 1.0:
        log_data.append([
            datetime.now().strftime("%H:%M:%S"),
            bpm_text,
            int(current_rmssd),
            stress_level,
            current_emotion,
            "DROWSY" if drowsy_alert else "AWAKE"
        ])
        last_log_time = time.time()

    cv2.imshow("Multimodal Health Monitor", frame)
    frame_count += 1
    if cv2.waitKey(1) & 0xFF == ord('q'): break

# Exit Cleanup
stop_threads = True
t.join()
cap.release()
cv2.destroyAllWindows()
plt.ioff()

print("Saving Data...")
with open("complete_health_log.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Time", "BPM", "HRV", "Stress", "Emotion", "Status"])
    writer.writerows(log_data)
print("Saved! Exiting.")
plt.show()