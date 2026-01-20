import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)
green_signal = []

FOREHEAD_POINTS = [10, 67, 69, 104, 108, 109]
LEFT_CHEEK_POINTS = [50, 101, 118, 119, 120, 47]
RIGHT_CHEEK_POINTS = [280, 330, 347, 348, 349, 277]

def extract_roi(frame, landmarks, indices, w, h):
    xs = [int(landmarks[i].x * w) for i in indices]
    ys = [int(landmarks[i].y * h) for i in indices]

    x1, x2 = max(min(xs), 0), min(max(xs), w)
    y1, y2 = max(min(ys), 0), min(max(ys), h)

    if x2 <= x1 or y2 <= y1:
        return None, None

    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    frame.flags.writeable = False
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)
    frame.flags.writeable = True

    if result.multi_face_landmarks:
        landmarks = result.multi_face_landmarks[0].landmark

        roi_values = []

       
        fh, box = extract_roi(frame, landmarks, FOREHEAD_POINTS, w, h)
        if fh is not None:
            roi_values.append(np.mean(fh[:, :, 1]))
            x1,y1,x2,y2 = box
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame,"Forehead",(x1,y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)

      
        lc, box = extract_roi(frame, landmarks, LEFT_CHEEK_POINTS, w, h)
        if lc is not None:
            roi_values.append(np.mean(lc[:, :, 1]))
            x1,y1,x2,y2 = box
            cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
            cv2.putText(frame,"Left Cheek",(x1,y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),1)

       
        rc, box = extract_roi(frame, landmarks, RIGHT_CHEEK_POINTS, w, h)
        if rc is not None:
            roi_values.append(np.mean(rc[:, :, 1]))
            x1,y1,x2,y2 = box
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
            cv2.putText(frame,"Right Cheek",(x1,y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1)

        if len(roi_values) > 0:
            green_signal.append(np.mean(roi_values))

    cv2.imshow("Stable Multi‑ROI rPPG", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

np.savetxt("green_signal.txt", green_signal)
print("Total frames captured:", len(green_signal))
