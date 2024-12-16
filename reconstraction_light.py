#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import warnings
from datetime import datetime


import torch
from tqdm import tqdm

import warnings

import rasterio
from rasterio.transform import from_origin


ap = argparse.ArgumentParser()
ap.add_argument("-img", "--input_image", default="./input_n/", required = False,
    help = "Path to the directory that contains the input imags")
ap.add_argument("-out", "--output_image", default="./output/", required = False,
    help = "Path to the directory that contains the output imags")


args = vars(ap.parse_args())



import logging

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, 
            format='%(asctime)s - %(levelname)s - %(message)s', 
            datefmt='%Y-%m-%d %H:%M:%S')

class Reconstraction(object):
    def __init__(self):
        self.path = args["input_image"]
        self.resize_coef = 0.1

    def __read_files(self, path = None):
        if path is None:
            path = self.path
        return os.listdir(path)

    def main(self):
        image_list = self.__read_files()
        print(f"[info] img list: {image_list}")
        self.projection(img1_path = image_list[1], img2_path = image_list[0])

    def projection(self, img1_path, img2_path):

        # Загрузка изображений
        img1 = cv2.imread(self.path + img1_path, cv2.IMREAD_GRAYSCALE)  # Изображение 1
        img2 = cv2.imread(self.path + img2_path, cv2.IMREAD_GRAYSCALE)  # Изображение 2

        img1 = cv2.resize(img1, (0, 0), fx = self.resize_coef, fy = self.resize_coef)
        img2 = cv2.resize(img2, (0, 0), fx = self.resize_coef, fy = 0.09)


        # Инициализация детектора SIFT
        sift = cv2.SIFT_create()

        # Извлечение ключевых точек и дескрипторов
        keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
        keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

        # Проверка наличия дескрипторов
        if descriptors1 is None or descriptors2 is None or len(keypoints1) == 0 or len(keypoints2) == 0:
            print("Не удалось извлечь ключевые точки или дескрипторы на одном из изображений.")
            exit()

        # Используем BFMatcher для поиска соответствий между ключевыми точками
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

        # Поиск соответствий
        matches = bf.match(descriptors1, descriptors2)

        # Сортировка соответствий по расстоянию
        matches = sorted(matches, key=lambda x: x.distance)

        # Создание списков точек для матрицы гомографии
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Вычисление матрицы гомографии
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Размеры второго изображения (куда будем проецировать img1)
        h2, w2 = img2.shape

        # Проецирование первого изображения (img1) на второе (img2) с использованием гомографии
        img1_warped = cv2.warpPerspective(img1, M, (w2, h2))

        # Маска для наложения
        mask_warped = cv2.warpPerspective(np.ones_like(img1, dtype=np.uint8) * 255, M, (w2, h2))

        # Объединение изображений
        img2_with_projection = cv2.bitwise_and(img2, cv2.bitwise_not(mask_warped))  # Убираем наложенную область с img2
        result = cv2.add(img2_with_projection, img1_warped)  # Добавляем наложенное изображение img1_warped

        # Найти углы изображения img1 для создания прямоугольника
        h1, w1 = img1.shape
        corners_img1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
        transformed_corners = cv2.perspectiveTransform(corners_img1, M)

        # Нарисовать синий прямоугольник вокруг проецированного изображения
        transformed_corners = np.int32(transformed_corners)  # Преобразование в целые числа
        cv2.polylines(result, [transformed_corners], isClosed=True, color=(255, 0, 0), thickness=5)

        # Отображение результата
        cv2.imshow('Projected Full Image', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    pm = Reconstraction()
    pm.main()