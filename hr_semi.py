import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch

# ----------------------------
# Parameters
# ----------------------------
fs = 30                         # Webcam FPS (Avg)
window_size = fs * 10           # 10-second window
BUFFER_MAX = fs * 15            # Keep max 15s of data
SMOOTHING_WINDOW = 5            # BPM averaging window

# State Variables
green_signal = []
raw_bpm = []
smooth_bpm = []
frame_idx = []
frame_count = 0
last_valid_bpm = None

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

# Stable ROI Landmarks (Forehead & Cheeks)
FOREHEAD = [10, 67, 69, 104, 108, 109]
LEFT_CHEEK = [50, 101, 118, 119, 120, 47]
RIGHT_CHEEK = [280, 330, 347, 348, 349, 277]

def extract_roi(frame, lm, idxs, w, h):
    xs = [int(lm[i].x * w) for i in idxs]
    ys = [int(lm[i].y * h) for i in idxs]
    x1, x2 = max(min(xs), 0), min(max(xs), w)
    y1, y2 = max(min(ys), 0), min(max(ys), h)
    
    if x2 <= x1 or y2 <= y1:
        return None, None
        
    roi = frame[y1:y2, x1:x2]
    return roi, (x1, y1, x2, y2)

# ----------------------------
# Plot Setup (The ONLY place results will show)
# ----------------------------
plt.ion()
fig, ax = plt.subplots()
line, = ax.plot([], [], 'r-', linewidth=2)
ax.set_xlabel("Frame Number")
ax.set_ylabel("Heart Rate (BPM)")
ax.set_title("Real-Time Heart Rate")
ax.set_ylim(40, 120)  # Human range
ax.grid(True)

cap = cv2.VideoCapture(0)

# ----------------------------
# Main Loop
# ----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    frame.flags.writeable = False
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)
    frame.flags.writeable = True

    face_detected = False
    current_bpm = 0

    if res.multi_face_landmarks:
        face_detected = True
        lm = res.multi_face_landmarks[0].landmark
        roi_vals = []

        # Draw ROIs (Boxes only, no text)
        for region, color in zip([FOREHEAD, LEFT_CHEEK, RIGHT_CHEEK], 
                                 [(0,255,0), (255,0,0), (0,0,255)]):
            roi, box = extract_roi(frame, lm, region, w, h)
            if roi is not None:
                roi_vals.append(np.mean(roi[:, :, 1]))
                # Optional: Comment out the line below if you want a clean video feed
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 1)

        # Buffer Logic
        if len(roi_vals) >= 2:
            green_signal.append(np.mean(roi_vals))
            if len(green_signal) > BUFFER_MAX:
                green_signal.pop(0)
    else:
        # Reset if face lost
        green_signal = [] 
        last_valid_bpm = None
        face_detected = False

    # ----------------------------
    # Signal Processing
    # ----------------------------
    if face_detected and len(green_signal) >= window_size:
        signal = np.array(green_signal[-window_size:])
        signal = signal - np.mean(signal)

        # Bandpass
        nyquist = fs / 2
        b, a = butter(3, [0.75/nyquist, 3.0/nyquist], btype='band')
        filtered = filtfilt(b, a, signal)

        freqs, psd = welch(filtered, fs, nperseg=len(filtered), nfft=2048)
        
        valid_idxs = np.where((freqs >= 0.75) & (freqs <= 3.0))
        valid_psd = psd[valid_idxs]
        valid_freqs = freqs[valid_idxs]

        if len(valid_psd) > 0:
            peak_idx = np.argmax(valid_psd)
            raw_val = valid_freqs[peak_idx] * 60
            
    
            if last_valid_bpm is None:
                last_valid_bpm = raw_val
            else:
                last_valid_bpm = (0.2 * raw_val) + (0.8 * last_valid_bpm)
            
            current_bpm = last_valid_bpm

   
    if face_detected:
        smooth_bpm.append(current_bpm if current_bpm > 0 else 0)
    else:
        smooth_bpm.append(0)

    frame_idx.append(frame_count)

    if len(frame_idx) > 200:
        frame_idx.pop(0)
        smooth_bpm.pop(0)

    line.set_xdata(frame_idx)
    line.set_ydata(smooth_bpm)
    ax.set_xlim(max(0, frame_count - 150), frame_count + 10)
    plt.pause(0.001)

   
    cv2.imshow("Face View", frame)
    
    frame_count += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.show() 