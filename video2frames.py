import cv2
import os

# Путь к видеофайлу
video_path = './150-5.mp4'

# Директория для сохранения кадров
output_dir = './img14/'
os.makedirs(output_dir, exist_ok=True)

# Открытие видео
cap = cv2.VideoCapture(video_path)

# Проверка, удалось ли открыть видео
if not cap.isOpened():
    print("Error: Не удалось открыть видео.")
else:
    frame_count = 0
    while True:
        # Чтение очередного кадра
        ret, frame = cap.read()
        
        # Если кадр не получен, значит, видео закончено
        if not ret:
            break
        
        # Сохранение кадра
        frame_filename = os.path.join(output_dir, f'frame_{frame_count:04d}.jpg')
        cv2.imwrite(frame_filename, frame)
        
        # Переход к следующему кадру
        frame_count += 1
        print(f"Сохранён {frame_filename}")
    
    # Освобождаем ресурсы
    cap.release()
    print(f"Всего сохранено {frame_count} кадров.")