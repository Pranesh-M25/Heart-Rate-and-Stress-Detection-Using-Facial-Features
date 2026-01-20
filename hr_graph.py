import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch

# ----------------------------
# Parameters
# ----------------------------
fs = 30
window_size = fs * 10
BUFFER_MAX = fs * 15
SMOOTH_ALPHA = 0.2
MIN_BPM = 45
MAX_BPM = 120
MIN_SIGNAL_STD = 0.4

RMSSD_WINDOW = 20

# ----------------------------
# State Variables
# ----------------------------
green_signal = []
smooth_bpm = []
frame_idx = []
rr_intervals = []

frame_count = 0
last_valid_bpm = None
current_rmssd = 0

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

# ROI landmarks
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
    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)

# ----------------------------
# Plot Setup
# ----------------------------
plt.ion()
fig, ax = plt.subplots()
line, = ax.plot([], [], 'r-', linewidth=2)
ax.set_xlabel("Frame Number")
ax.set_ylabel("Heart Rate (BPM)")
ax.set_title("Real-Time Heart Rate & HRV")
ax.set_ylim(MIN_BPM, MAX_BPM)
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
    current_bpm = None

    if res.multi_face_landmarks:
        face_detected = True
        lm = res.multi_face_landmarks[0].landmark
        roi_vals = []

        for region, color in zip(
            [FOREHEAD, LEFT_CHEEK, RIGHT_CHEEK],
            [(0,255,0), (255,0,0), (0,0,255)]
        ):
            roi, box = extract_roi(frame, lm, region, w, h)
            if roi is not None:
                roi_vals.append(np.mean(roi[:, :, 1]))
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 1)

        if len(roi_vals) >= 2:
            green_signal.append(np.mean(roi_vals))
            if len(green_signal) > BUFFER_MAX:
                green_signal.pop(0)

    else:
        green_signal.clear()
        rr_intervals.clear()
        last_valid_bpm = None
        current_rmssd = 0

    # ----------------------------
    # Heart Rate Estimation
    # ----------------------------
    if face_detected and len(green_signal) >= window_size:
        signal = np.array(green_signal[-window_size:])
        signal -= np.mean(signal)

        b, a = butter(3, [0.75/(fs/2), 3.0/(fs/2)], btype='band')
        filtered = filtfilt(b, a, signal)

        if np.std(filtered) >= MIN_SIGNAL_STD:
            freqs, psd = welch(filtered, fs, nperseg=len(filtered), nfft=2048)
            valid = np.where((freqs >= 0.75) & (freqs <= 3.0))

            if len(valid[0]) > 0:
                raw_bpm = freqs[valid][np.argmax(psd[valid])] * 60

                if last_valid_bpm is None:
                    last_valid_bpm = raw_bpm
                else:
                    last_valid_bpm = (
                        SMOOTH_ALPHA * raw_bpm +
                        (1 - SMOOTH_ALPHA) * last_valid_bpm
                    )

                current_bpm = np.clip(last_valid_bpm, MIN_BPM, MAX_BPM)

                # -------- HRV (RMSSD) --------
                rr = 60000 / current_bpm
                rr_intervals.append(rr)
                if len(rr_intervals) > RMSSD_WINDOW:
                    rr_intervals.pop(0)

                if len(rr_intervals) >= 2:
                    diff_rr = np.diff(rr_intervals)
                    current_rmssd = np.sqrt(np.mean(diff_rr ** 2))

    # ----------------------------
    # Update Graph
    # ----------------------------
    smooth_bpm.append(current_bpm if current_bpm else 0)
    frame_idx.append(frame_count)

    if len(frame_idx) > 200:
        frame_idx.pop(0)
        smooth_bpm.pop(0)

    line.set_xdata(frame_idx)
    line.set_ydata(smooth_bpm)
    ax.set_xlim(max(0, frame_count - 150), frame_count + 10)
    plt.pause(0.001)

    # ----------------------------
    # Display
    # ----------------------------
    cv2.putText(
        frame,
        f"HRV (RMSSD): {int(current_rmssd)} ms",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2
    )

    cv2.imshow("Face View", frame)

    frame_count += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ----------------------------
# Cleanup
# ----------------------------
cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.show()
