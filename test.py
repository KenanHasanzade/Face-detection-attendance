import cv2
import face_recognition
import os
import pickle
import numpy as np
import threading
import time

# --- НАСТРОЙКИ ---
RTSP_URL = "rtsp://admin:Admin123@172.30.42.242:554/Streaming/Channels/101"
DATABASE_FILE = "encodings.pickle"
DISPLAY_WIDTH, DISPLAY_HEIGHT = 1280, 720
PROCESS_EVERY_N_FRAME = 3 
USE_GPU = True

class VideoStream:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        self.ret, self.frame = self.cap.read()
        self.stopped = False
        self.lock = threading.Lock() # Защита для передачи кадра между потоками

    def start(self):
        t = threading.Thread(target=self.update, args=(), daemon=True)
        t.start()
        return self

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                continue
            with self.lock:
                self.frame = frame

    def read(self):
        with self.lock:
            # Возвращаем копию, чтобы не было конфликтов при отрисовке
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.stopped = True
        self.cap.release()

def main():
    if not os.path.exists(DATABASE_FILE):
        print("[ERROR] Сначала создай базу через скрипт кодирования!")
        return

    with open(DATABASE_FILE, "rb") as f:
        data = pickle.load(f)

    # Запускаем поток захвата
    vs = VideoStream(RTSP_URL).start()
    
    # Ждем появления первого кадра
    print("[INFO] Waiting for camera stream...")
    while vs.read() is None:
        time.sleep(0.1)

    cv2.namedWindow('Jetson Ultra Fast FaceID', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Jetson Ultra Fast FaceID', DISPLAY_WIDTH, DISPLAY_HEIGHT)

    face_locations = []
    face_names = []
    frame_count = 0
    prev_time = 0

    print("[INFO] Engine started. Processing...")

    while True:
        frame = vs.read()
        if frame is None:
            continue

        # Считаем FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        # 1. Сжатие для детектора (0.2 = в 5 раз меньше)
        small_frame = cv2.resize(frame, (0, 0), fx=0.2, fy=0.2)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # 2. Детекция только в каждом N-м кадре
        if frame_count % PROCESS_EVERY_N_FRAME == 0:
            model_type = "cnn" if USE_GPU else "hog"
            face_locations = face_recognition.face_locations(rgb_small_frame, model=model_type)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(data["encodings"], face_encoding, tolerance=0.5)
                name = "Unknown"
                
                face_distances = face_recognition.face_distance(data["encodings"], face_encoding)
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = data["names"][best_match_index]
                face_names.append(name)

        frame_count += 1

        # 3. Отрисовка
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Масштабируем координаты обратно (0.2 -> 5.0)
            top *= 5; right *= 5; bottom *= 5; left *= 5
            
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Пишем FPS на экране
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # 4. Вывод
        output_frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
        cv2.imshow('Jetson Ultra Fast FaceID', output_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vs.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
