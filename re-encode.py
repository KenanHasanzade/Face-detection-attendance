import face_recognition
import os
import pickle
import sys

EMPLOYEES_FOLDER = "employees/"
DATABASE_FILE = "encodings.pickle"

def refresh_database():
    print(f"[INFO] Scanning folder: {EMPLOYEES_FOLDER}...")
    known_encodings = []
    known_names = []

    if not os.path.exists(EMPLOYEES_FOLDER):
        print(f"[ERROR] Folder '{EMPLOYEES_FOLDER}' not found!")
        return

    for file in os.listdir(EMPLOYEES_FOLDER):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(EMPLOYEES_FOLDER, file)
            print(f"[PROCESS] Encoding: {file}...")
            
            image = face_recognition.load_image_file(path)
            # Используем больше ресурсов для более точного кодирования в базу
            encodings = face_recognition.face_encodings(image, num_jitters=1)
            
            if len(encodings) > 0:
                known_encodings.append(encodings[0])
                known_names.append(os.path.splitext(file)[0])
                print(f"[SUCCESS] Done!")
            else:
                print(f"[WARNING] Face not found in {file}! Skipping.")

    # Сохраняем (перезаписываем) файл
    data = {"encodings": known_encodings, "names": known_names}
    with open(DATABASE_FILE, "wb") as f:
        f.write(pickle.dumps(data))
    
    print(f"\n[FINISH] Database updated! Total employees: {len(known_names)}")

if __name__ == "__main__":
    refresh_database()