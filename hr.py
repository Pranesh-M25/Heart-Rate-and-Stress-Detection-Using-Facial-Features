import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch, find_peaks

# ----------------------------
# Parameters
# ----------------------------
fs = 30                          # Webcam FPS (Approx)
window_size = fs * 10            # 10-second analysis window
BUFFER_MAX = fs * 15             # 15-second buffer limit
SMOOTH_ALPHA = 0.2               # BPM smoothing factor

# Stress Thresholds (ms)
STRESS_HIGH_THRESH = 25          # High stress
STRESS_LOW_THRESH = 50           # Relaxed

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

    roi = frame[y1:y2, x1:x2]
    return np.mean(roi[:, :, 1]), (x1, y1, x2, y2)

# ----------------------------
# Matplotlib Setup
# ----------------------------
plt.ion()
fig, ax = plt.subplots()
line, = ax.plot([], [], linewidth=2)
ax.set_xlabel("Time (Frames)")
ax.set_ylabel("Heart Rate (BPM)")
ax.set_title("Real-Time Stress & Heart Rate Monitor")
ax.set_ylim(40, 140)
ax.grid(True)

print("Starting System... Press 'q' to exit.")

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

    # ----------------------------
    # 1. Face Detection & ROI Extraction
    # ----------------------------
    if res.multi_face_landmarks:
        face_detected = True
        lm = res.multi_face_landmarks[0].landmark
        roi_means = []

        for region, color in zip(
            [FOREHEAD, LEFT_CHEEK, RIGHT_CHEEK],
            [(0,255,0), (255,0,0), (0,0,255)]
        ):
            val, box = extract_roi(frame, lm, region, w, h)
            if val is not None:
                roi_means.append(val)
                cv2.rectangle(frame, box[:2], box[2:], color, 1)

        if len(roi_means) >= 2:
            green_signal.append(np.mean(roi_means))
            if len(green_signal) > BUFFER_MAX:
                green_signal.pop(0)
    else:
        green_signal.clear()
        last_valid_bpm = None
        current_rmssd = 0
        stress_level = "NO FACE"
        stress_color = (100, 100, 100)

    # ----------------------------
    # 2. Signal Processing
    # ----------------------------
    if face_detected and len(green_signal) >= window_size:
        raw_data = np.array(green_signal[-window_size:])
        normalized = raw_data - np.mean(raw_data)

        nyquist = fs / 2
        b, a = butter(3, [0.7/nyquist, 3.5/nyquist], btype='band')
        filtered = filtfilt(b, a, normalized)

        # ---- Heart Rate (Frequency Domain)
        freqs, psd = welch(filtered, fs, nperseg=len(filtered), nfft=2048)
        valid = (freqs >= 0.7) & (freqs <= 3.5)

        if np.any(valid):
            raw_bpm = freqs[valid][np.argmax(psd[valid])] * 60
            if last_valid_bpm is None:
                last_valid_bpm = raw_bpm
            else:
                last_valid_bpm = (
                    SMOOTH_ALPHA * raw_bpm +
                    (1 - SMOOTH_ALPHA) * last_valid_bpm
                )

        # ---- HRV (RMSSD)
        peaks, _ = find_peaks(filtered, distance=int(fs * 0.4))

        if len(peaks) > 2:
            rr_ms = np.diff(peaks) / fs * 1000
            diff_rr = np.diff(rr_ms)

            if len(diff_rr) > 0:
                current_rmssd = np.sqrt(np.mean(diff_rr ** 2))

            if current_rmssd < STRESS_HIGH_THRESH:
                stress_level = "HIGH STRESS"
                stress_color = (0, 0, 255)
                plot_color = "red"
            elif current_rmssd < STRESS_LOW_THRESH:
                stress_level = "MEDIUM STRESS"
                stress_color = (0, 255, 255)
                plot_color = "orange"
            else:
                stress_level = "RELAXED"
                stress_color = (0, 255, 0)
                plot_color = "green"
        else:
            plot_color = "gray"
    else:
        plot_color = "gray"
        if face_detected:
            stress_level = "CALIBRATING..."
            stress_color = (255, 255, 0)

    # ----------------------------
    # 3. Plot Update
    # ----------------------------
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

    # ----------------------------
    # 4. UI Overlay
    # ----------------------------
    cv2.rectangle(frame, (0, 0), (360, 160), (0, 0, 0), -1)

    cv2.putText(frame, f"BPM: {int(last_valid_bpm) if last_valid_bpm else '--'}",
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.putText(frame, f"HRV (RMSSD): {int(current_rmssd)} ms",
                (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)

    cv2.putText(frame, stress_level,
                (20, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.9, stress_color, 2)

    cv2.imshow("Heart Rate & Stress Detector", frame)
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.show()
