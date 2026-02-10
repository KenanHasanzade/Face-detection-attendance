import cv2
import mediapipe as mp
import face_recognition
import numpy as np
import pickle
import threading
import time
import os

# --- НАСТРОЙКИ ---
RTSP_URL = "rtsp://admin:Admin123@172.30.42.242:554/Streaming/Channels/101"
DATABASE_FILE = "encodings.pickle"

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(
    model_selection=1, 
    min_detection_confidence=0.5
)
class VideoStream:
    def __init__(self, src):
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|low_delay;1"
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.frame = None
        self.stopped = False
        self.lock = threading.Lock()

    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            if not self.cap.grab(): continue
            ret, frame = self.cap.retrieve()
            if ret:
                with self.lock:
                    self.frame = frame

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.stopped = True
        self.cap.release()

def main():
    with open(DATABASE_FILE, "rb") as f:
        data = pickle.load(f)

    vs = VideoStream(RTSP_URL).start()
    time.sleep(2.0)

    print("[INFO] MediaPipe Engine Started. Visuals Updated.")

    while True:
        frame = vs.read()
        if frame is None: continue

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)

        if results.detections:
            face_locations = []
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                top = int(bbox.ymin * h)
                left = int(bbox.xmin * w)
                bottom = int((bbox.ymin + bbox.height) * h)
                right = int((bbox.xmin + bbox.width) * w)
                
                top, left = max(0, top), max(0, left)
                bottom, right = min(h, bottom), min(w, right)
                face_locations.append((top, right, bottom, left))

            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(data["encodings"], face_encoding, tolerance=0.5)
                name = "Unknown"
                face_distances = face_recognition.face_distance(data["encodings"], face_encoding)
                
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = data["names"][best_match_index]

                # Отрисовка: Красный для Unknown, Зеленый для своих
                color = (0, 0, 255) if name == "Unknown" else (0, 255, 0)
                
                # Рамка и имя (крупно)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 3)
                cv2.putText(frame, name, (left, top - 15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        cv2.imshow("Jetson Fast MediaPipe", cv2.resize(frame, (1280, 720)))
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    vs.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()