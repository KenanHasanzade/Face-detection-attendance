import cv2
import numpy as np
import face_recognition
import pickle
import threading
import time
import os
import sys

# НАСТРОЙКИ ПОДКЛЮЧЕНИЯ К КАМЕРЕ
RTSP_URL = "rtsp://admin:Admin123@172.30.42.242:554/Streaming/Channels/101"
DATABASE_FILE = "encodings.pickle"

class VideoStream:
    def __init__(self, src):
        # Оптимальные параметры для Jetson: TCP транспорт и минимальный буфер
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|low_delay;1"
        self.cap = cv2.VideoCapture(src)
        
        # Получаем родное разрешение камеры
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
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

# Глобальные данные для потока распознавания
face_locations = []
face_names = []

def face_recognition_thread(vs, data):
    global face_locations, face_names
    # Коэффициент сжатия ТОЛЬКО для анализа (не влияет на экран)
    scale = 0.2 
    
    while not vs.stopped:
        frame = vs.read()
        if frame is None:
            time.sleep(0.01)
            continue
        
        # Для анализа сжимаем сильно, чтобы не тормозил FPS
        small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Детекция на маленьком кадре
        current_locations = face_recognition.face_locations(rgb_small, model="hog")
        current_names = []
        
        if current_locations:
            current_encodings = face_recognition.face_encodings(rgb_small, current_locations)
            for encoding in current_encodings:
                matches = face_recognition.compare_faces(data["encodings"], encoding, tolerance=0.5)
                name = "Unknown"
                dist = face_recognition.face_distance(data["encodings"], encoding)
                if len(dist) > 0 and matches[np.argmin(dist)]:
                    name = data["names"][np.argmin(dist)]
                current_names.append(name)
        
        face_locations = current_locations
        face_names = current_names
        time.sleep(0.03)

def main():
    if not os.path.exists(DATABASE_FILE):
        print("[ERROR] Database file not found!")
        return

    with open(DATABASE_FILE, "rb") as f:
        data = pickle.load(f)

    vs = VideoStream(RTSP_URL).start()
    time.sleep(2.0)

    # Запуск распознавания в фоне
    threading.Thread(target=face_recognition_thread, args=(vs, data), daemon=True).start()

    print(f"[INFO] Crystal Clear Mode: {vs.width}x{vs.height}")
    prev_frame_time = time.time()

    # Создаем окно с возможностью изменения размера, но без потери качества
    cv2.namedWindow("Jetson High Quality", cv2.WINDOW_NORMAL)

    while True:
        frame = vs.read()
        if frame is None: continue

        # ОТРИСОВКА на ОРИГИНАЛЬНОМ кадре
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Масштабируем координаты обратно к оригиналу (1 / 0.2 = 5)
            s = 5
            t, r, b, l = top * s, right * s, bottom * s, left * s
            
            # Адаптивная толщина линий в зависимости от разрешения
            thickness = max(2, int(vs.width / 500))
            
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (l, t), (r, b), color, thickness)
            cv2.putText(frame, name, (l, t - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, vs.width/1500, color, thickness)

        # Рассчет и вывод FPS на экран
        curr_time = time.time()
        fps = 1 / (curr_time - prev_frame_time)
        prev_frame_time = curr_time

        # Выводим инфо
        cv2.putText(frame, f"FPS: {int(fps)} | Res: {vs.width}x{vs.height}", (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, vs.width/1200, (0, 255, 0), 2)

        # ПОКАЗ ОРИГИНАЛА(без cv2.resize)
        cv2.imshow("Jetson High Quality", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

    vs.stopped = True
    cv2.destroyAllWindows()
    os._exit(0)

if __name__ == "__main__":
    main()
