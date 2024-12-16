import json
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor
import os
import calculate_utils
import pandas as pd
import simplekml
import zipfile
import tempfile
from fastkml import kml
from shapely.geometry import Point

'''
# Загрузка JSON
with open("./img15_1/1_layer/1/2024_12_02_23_41_59.370552res.json", "r") as file:
    data = json.load(file)

# Список изображений
images = data["images"].split()

# Чтение матриц гомографии
raw_homographies = data["homography"]

# Убираем лишние символы и парсим матрицы
homographies = []
for h in raw_homographies.strip('[]').split('], ['):
    h = h.replace('\n', '').replace('[', '').replace(']', '')
    matrix_flat = list(map(float, h.split()))
    homographies.append(np.array(matrix_flat, dtype=np.float32).reshape(3, 3))

# Размеры изображений (заменить на реальные размеры)
image_shapes = [(2000, 1500)] * len(images)  # (height, width)

# Начальная матрица (единичная) для аккумулирования
accumulated_homographies = [np.eye(3, dtype=np.float32)]

# Вычисляем аккумулированные матрицы гомографии
for H in homographies:
    accumulated_homographies.append(accumulated_homographies[-1] @ H)

# Вычисляем границы ортофотоплана
all_corners = []
for H_acc, (h, w) in zip(accumulated_homographies, image_shapes):
    corners = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]]).T
    transformed_corners = H_acc @ corners
    transformed_corners /= transformed_corners[2]  # Нормализация
    all_corners.extend(transformed_corners[:2].T)

all_corners = np.vstack(all_corners)
x_min, y_min = all_corners.min(axis=0)
x_max, y_max = all_corners.max(axis=0)

# Размеры ортофотоплана
ortho_width = int(x_max - x_min)
ortho_height = int(y_max - y_min)

# Смещение для перевода координат в положительный диапазон
translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float32)

# Пустой ортофотоплан (TIFF)
orthoimage = np.zeros((ortho_height, ortho_width, 3), dtype=np.uint8)

# Функция для вычисления центра и преобразования
def calculate_center(H_acc, image_shape, image_name):
    h, w = image_shape
    center = np.array([w / 2, h / 2, 1], dtype=np.float32).T
    transformed_center = translation @ H_acc @ center
    transformed_center /= transformed_center[2]  # Нормализация
    return (transformed_center[:2], image_name)

# Параллельное выполнение вычислений
with ThreadPoolExecutor() as executor:
    centers = list(executor.map(calculate_center, accumulated_homographies, image_shapes, images))

#print()

# Отрисовка центров и названий
#for center, img_name in centers:
#    cx, cy = map(int, center)
#    # Рисуем круг в центре
#    cv2.circle(orthoimage, (orthoimage.shape[1] - cx, orthoimage.shape[0] - cy), radius=10, color=(0, 255, 0), thickness=-1)
#    # Добавляем текст с названием изображения
#    cv2.putText(orthoimage, img_name, (orthoimage.shape[1] - cx + 15, orthoimage.shape[0] - cy - 15), 
#                fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
#                fontScale=0.5, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)


# Сохранение результата в TIFF
#cv2.imwrite("orthoimage_with_centers.tiff", orthoimage)
#print("Ортофотоплан с центрами сохранён в 'orthoimage_with_centers.tiff'")

'''
def calculate_rotation_angle(H):
    """
    Вычисляет угол поворота в градусах из матрицы гомографии.
    
    Parameters:
        H (numpy.ndarray): Матрица гомографии размером 3x3.
    
    Returns:
        float: Угол поворота в градусах.
    """
    # Извлечение элементов матрицы, связанных с вращением
    h11, h12, h13 = H[0]
    h21, h22, h23 = H[1]
    
    # Нормализация, чтобы избавиться от масштаба (опционально)
    scale = np.sqrt(h11**2 + h21**2)
    h11 /= scale
    h21 /= scale
    
    # Вычисление угла поворота
    theta = np.arctan2(h21, h11)
    angle_degrees = np.degrees(theta)
    
    return angle_degrees



def core(json_file, wear):
    try:
        # Загрузка JSON
        with open("./img15_1/1_layer/1/" + json_file, "r") as file:
            data = json.load(file)

        # Список изображений
        images = data["images"].split()

        # Чтение матриц гомографии
        raw_homographies = data["homography"]

        # Убираем лишние символы и парсим матрицы
        homographies = []
        for h in raw_homographies.strip('[]').split('], ['):
            h = h.replace('\n', '').replace('[', '').replace(']', '')
            matrix_flat = list(map(float, h.split()))
            homographies.append(np.array(matrix_flat, dtype=np.float32).reshape(3, 3))

        # Размеры изображений (заменить на реальные размеры)
        image_shapes = [(4700, 7300)] * 6 #len(images)  # (height, width)

        # Начальная матрица (единичная) для аккумулирования
        accumulated_homographies = [np.eye(3, dtype=np.float32)]

        # Вычисляем аккумулированные матрицы гомографии
        for H in homographies:
            accumulated_homographies.append(accumulated_homographies[-1] @ H)

        # Вычисляем границы ортофотоплана
        all_corners = []
        for H_acc, (h, w) in zip(accumulated_homographies, image_shapes):
            corners = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]]).T
            transformed_corners = H_acc @ corners
            transformed_corners /= transformed_corners[2]  # Нормализация
            all_corners.extend(transformed_corners[:2].T)

        all_corners = np.vstack(all_corners)
        x_min, y_min = all_corners.min(axis=0)
        x_max, y_max = all_corners.max(axis=0)

        # Размеры ортофотоплана
        ortho_width = int(x_max - x_min)
        ortho_height = int(y_max - y_min)

        # Смещение для перевода координат в положительный диапазон
        translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float32)

        # Пустой ортофотоплан (TIFF)
        orthoimage = np.zeros((ortho_height, ortho_width, 3), dtype=np.uint8)
        print("shape: ", orthoimage.shape)
        def calculate_center(H_acc, image_shape, image_name):
            h, w = image_shape
            center = np.array([w / 2, h / 2, 1], dtype=np.float32).T
            transformed_center = translation @ H_acc @ center
            transformed_center /= transformed_center[2]  # Нормализация
            return (transformed_center[:2], image_name)


        with ThreadPoolExecutor() as executor:
            centers = list(executor.map(calculate_center, accumulated_homographies, image_shapes, images))  

        '''
        for center, img_name in centers:
            cx, cy = map(int, center)
            print(f"{img_name} has coordinates: {orthoimage.shape[1] - cx}, {orthoimage.shape[0] - cy}")
            
            theta = wear + ..
            x, y = cu.Geodesy().polar_to_xy(r, theta, image_shape) '''

    except Exception as ex:
        print(">>> ", ex) 



def recover_image_centers(path, df):
    
    json_files = [f for f in os.listdir(path) if f.endswith('.json')]
    for i in range(len(json_files)):
        if i > 0:
            wear = df.iloc[i, "wear"]
        else:
            wear = 0
        core(json_file = json_files[i], wear = wear)






