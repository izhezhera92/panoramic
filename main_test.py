#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 15:01:43 2024

@author: jinxu
"""
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import warnings
from datetime import datetime
from math import cos, sin, pi, pow, sqrt, radians, atan2, degrees, tan

import torch
from tqdm import tqdm

from lightglue import LightGlue, SuperPoint, DISK, viz2d
from lightglue.utils import load_image, rbd
import warnings


import rasterio
from rasterio.transform import from_origin


ap = argparse.ArgumentParser()
ap.add_argument("-img", "--input_image", default="./img9_1/", required = False,
    help = "Path to the directory that contains the imags")
ap.add_argument("-out", "--output_image", default="res.tif", required = False,
    help = "Path to the output image")


args = vars(ap.parse_args())



import logging

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, 
            format='%(asctime)s - %(levelname)s - %(message)s', 
            datefmt='%Y-%m-%d %H:%M:%S')

class Panoranic_mode(object):
    def __init__(self, image_path = args["input_image"], verbose=True, 
        output_image = args["output_image"], descriptor_algo = "orb", 
        resize_coef = 1.0):
        self.image_path = image_path
        self.output_image = output_image
        self.verbose = verbose
        self.max_num_keypoints = 256
        self.device_type  = 'cpu' # 'mps'  
        self.device = torch.device("cuda" if torch.cuda.is_available() else self.device_type)
        self.resize_coef = resize_coef
        self.descriptor = "orb" #"orb", "lightGlue"
        self.nfeatures=1000
        self.descriptor_algo = descriptor_algo
        if self.descriptor_algo == "sift":
            self.descriptor_type = cv2.SIFT_create()  
        elif self.descriptor_algo == "orb":
            self.descriptor_type = cv2.ORB_create(nfeatures=self.nfeatures) 
        elif self.descriptor_algo =="brisk":
            self.descriptor_type = cv2.BRISK_create() 


        if torch.cuda.is_available():
            logging.info(f"Cude is available")
        else:
            logging.warning(f"Cude is not available")
        self.image_list_buffer = [None, None, None]
        self.size_edje = 20000
        self.red_centers = []
        self.blue_centers = []
        self.points = []



    def __matches_detection(self, image0, image1)-> tuple:
        ''' Method for the matches finding with LightGlue algorithm'''

        # setup SUperGlue detector
        extractor = SuperPoint(max_num_keypoints=self.max_num_keypoints).eval().to(self.device)  
        matcher = LightGlue(features="superpoint").eval().to(self.device)

        # finding points in both images
        feats0 = extractor.extract(image0.to(self.device))
        feats1 = extractor.extract(image1.to(self.device))

        # matching pairs
        matches01 = matcher({"image0": feats0, "image1": feats1})
        feats0, feats1, matches01 = [
            rbd(x) for x in [feats0, feats1, matches01]
        ]  # remove batch dimension

        kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
        m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
        #if self.verbose:
        #    logging.info(f"Detected: {len(matches)} pairs")
        return m_kpts0, m_kpts1, matches01, kpts0, kpts1

    def get_xy_by_course_and_distance(self, 
        x:int = 0, y:int = 0,
        l: float = 0.0,
        azimuth: float = 0.0)-> tuple:
        xx = 0
        yy = 0
        if x > 0 and y > 0:
            xx = x + (l * cos(azimuth))
            yy = y + (l * sin(azimuth))
        return xx, yy

    def calculate_distance_xy(self, x1, y1, x2, y2):
        """
        Method for dist calculation between 2 points in 2D
        input    -> p1, p2: (x1,y1), (x2,y2)
        output   -> dist 
        """
        if x1 and y1 and x2 and y2:
            return sqrt(pow(abs(x1 - x2), 2) + pow(abs(y1 - y2),2))

        else:
            return None

    def get_2d_course_angle_image_coord_sys(self, 
        x1:int = 0, y1:int = 0, 
        x2:int = 0, y2:int = 0
        ) -> float:
        """
        Method for 2d course angle calculation.
        Zero direction angle in the top of image, over on 1st point.
        Using image coordinate system.

        input  -> int: x1, y1, x2, y2
        output -> float: course_2d
        """

        print(x1, y1, x2, y2)
        x1 = x1 - x2
        y1 = y1 - y2
        x2, y2 = 0, 0

        angle = degrees(atan2(x2-x1, y2-y1))

        if angle <= 0 and angle >= -180:
            return -angle

        elif angle >= 0 and angle <= 180:
            return 360 - angle


    def warpImages(self, img1, img2, H):
        rows1, cols1 = img1.shape[:2]
        rows2, cols2 = img2.shape[:2]

        # Coordinates of the corners of the first image (reference)
        list_of_points_1 = np.float32([[0, 0], [0, rows1], [cols1, rows1], [cols1, 0]]).reshape(-1, 1, 2)
        # Coordinates of the corners of the second image
        temp_points = np.float32([[0, 0], [0, rows2], [cols2, rows2], [cols2, 0]]).reshape(-1, 1, 2)

        # Perspective transformation for the second image
        list_of_points_2 = cv2.perspectiveTransform(temp_points, H)

        # Concatenate the points of both images
        list_of_points = np.concatenate((list_of_points_1, list_of_points_2), axis=0)

        # Calculate the boundaries of the final stitched image
        [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)

        # Compute the translation to keep everything within view
        translation_dist = [-x_min, -y_min]
        H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

        # Warp the second image
        output_img = cv2.warpPerspective(img2, H_translation.dot(H), (x_max - x_min, y_max - y_min))

        # Paste the first image into the stitched result
        output_img[translation_dist[1]:rows1 + translation_dist[1], translation_dist[0]:cols1 + translation_dist[0]] = img1

        # Calculate the center points of each image and draw circles
        center_img1 = (cols1 // 2 + translation_dist[0], rows1 // 2 + translation_dist[1])  # Center of image 1

        # Correctly transform the center of image 2
        center_img2 = np.float32([[cols2 // 2, rows2 // 2]]).reshape(-1, 1, 2)  # Center of img2
        transformed_center_img2 = cv2.perspectiveTransform(center_img2, H_translation.dot(H))[0][0]
        
        new_left_top = np.float32([[0, 0]]).reshape(-1, 1, 2)  # new_left_bottom of img2
        transformed_new_left_top = cv2.perspectiveTransform(new_left_top, H_translation.dot(H))[0][0]

        new_left_bottom = np.float32([[0, rows2]]).reshape(-1, 1, 2)  # new_left_bottom of img2
        transformed_new_left_bottom = cv2.perspectiveTransform(new_left_bottom, H_translation.dot(H))[0][0]

        new_right_bottom = np.float32([[cols2, rows2]]).reshape(-1, 1, 2)  # new_right_bottom img 2
        transformed_new_right_bottom = cv2.perspectiveTransform(new_right_bottom, H_translation.dot(H))[0][0]

        new_right_top = np.float32([[cols2, 0]]).reshape(-1, 1, 2)  # new_right_top of img2
        transformed_new_right_top = cv2.perspectiveTransform(new_right_top, H_translation.dot(H))[0][0]


        

        # Draw circles at the centers of each image
        cv2.circle(output_img, (int(center_img1[0]), int(center_img1[1])), 10, (0, 0, 255), -1)  # Red circle for image 1
        cv2.circle(output_img, (int(transformed_center_img2[0]), int(transformed_center_img2[1])), 10, (255, 0, 0), -1)  # Blue circle for image 2
        print("[info] Printed blue and red")
        self.red_centers.append([int(center_img1[0]), int(center_img1[1])])
        self.blue_centers.append([int(transformed_center_img2[0]), int(transformed_center_img2[1])])
        print("blue center: ", int(center_img1[0]), int(center_img1[1]))
        print("red center: ", int(transformed_center_img2[0]), int(transformed_center_img2[1]))
        print("result image shape: ", output_img.shape)

        if not self.points:
            self.points.append([[int(center_img1[0]), int(center_img1[1])], [], [], [], []])

        self.points.append([[int(transformed_center_img2[0]), int(transformed_center_img2[1])],
            [int(transformed_new_left_top[0]), int(transformed_new_left_top[1])],
            [int(transformed_new_left_bottom[0]), int(transformed_new_left_bottom[1])],
            [int(transformed_new_right_bottom[0]), int(transformed_new_right_bottom[1])],
            [int(transformed_new_right_top[0]), int(transformed_new_right_top[1])]]
            )
        

        # Return the result image along with the centers
        return output_img, (center_img1, transformed_center_img2)

    def __read_files(self, path):
        return os.listdir(path)

    def __create_image_list(self, lst):
        img_list = []
        for i in range(len(lst)):
            new_image = cv2.imread(self.image_path + lst[i])
            new_image = cv2.resize(new_image, (0, 0), fx = self.resize_coef, fy = self.resize_coef)
            img_list.append(new_image)
        return img_list

    def __add_alpha_chanel(self, img):
        if img is not None:
            if img.shape[2] == 3:
                # Convert the 3-channel BGR to 4-channel BGRA by adding an alpha channel
                alpha_channel = np.ones((img.shape[0], img.shape[1]), dtype=img.dtype) * 255  # Full opacity
                img = cv2.merge((img, alpha_channel))

            # Convert from OpenCV BGR(A) to RGB(A)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)

            # Define the transform (affine transformation) for georeferencing (identity matrix in this case)
            transform = from_origin(0, 0, 1, 1)  # Adjust based on your spatial reference

            return transform, img
        return None, None

    def __tiff_saving(self, img, transform, path = './output_image.tif'):
        with rasterio.open(
            path,
            'w',
            driver='GTiff',
            height=img.shape[0],
            width=img.shape[1],
            count=4,  # 4 channels (Red, Green, Blue, Alpha)
            dtype=img.dtype,
            transform=transform,
            crs='+proj=latlong'  # You can set your desired CRS here
        ) as dst:
            # Write each channel separately
            dst.write(img[:, :, 0], 1)  # Red
            dst.write(img[:, :, 1], 2)  # Green
            dst.write(img[:, :, 2], 3)  # Blue
            dst.write(img[:, :, 3], 4)  # Alpha

        print("Image saved successfully with alpha channel as a TIFF.")

        return 0

    def __insert_image_buffer(self, img_list):
        img_list.insert(0, self.image_list_buffer[0])
        img_list.insert(1, self.image_list_buffer[1])
        img_list.insert(2, self.image_list_buffer[2])
        return img_list

    def __update_image_buffer(self, img):
        self.image_list_buffer[2] = self.image_list_buffer[1]
        self.image_list_buffer[1] = self.image_list_buffer[0]
        self.image_list_buffer[0] = img
        return 0

    def stiching(self, img_list=[], minmaxcount=5, file_list=[]):
        centers = []  # List to store the center coordinates of all images
        angles = []
        distances = []
        i = 0
        shift = 0
        result = None

        while True:
            print("len of image list: ", len(img_list))
            if len(img_list) < 2:
                now = str(datetime.now()).replace(" ","_").replace(":","_").replace("-","_")
                try:
                    transform, img = self.__add_alpha_chanel(img=result)
                    image_path = str("./res/") + str(now) + self.output_image
                    self.__tiff_saving(img=img, transform=transform, path=image_path)
                except:
                    pass
                print("[warning] Stopped. Not enought images")
                break

            img1 = img_list.pop(0)
            img2 = img_list.pop(1)

            self.__update_image_buffer(img=img2)

            keypoints1, descriptors1 = self.descriptor_type.detectAndCompute(img1, None)
            keypoints2, descriptors2 = self.descriptor_type.detectAndCompute(img2, None)

            if descriptors1 is None or descriptors2 is None or len(keypoints1) == 0 or len(keypoints2) == 0:
                print("[warning] Stopped. Not enought descriptors")
                break

            matches = None
            if self.descriptor_algo == "sift":
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                search_params = dict(checks=50)
                flann = cv2.FlannBasedMatcher(index_params, search_params)
                matches = flann.knnMatch(descriptors1, descriptors2, k=2)
            elif self.descriptor_algo == "orb":
                bf = cv2.BFMatcher_create(cv2.NORM_HAMMING)
                matches = bf.knnMatch(descriptors1, descriptors2, k=2)
            elif self.descriptor_algo == "brisk":
                bf = cv2.BFMatcher_create(cv2.NORM_HAMMING)
                matches = bf.knnMatch(descriptors1, descriptors2, k=2)

            good = [m for m, n in matches if m.distance < 0.5 * n.distance]

            MIN_MATCH_COUNT = minmaxcount
            try:
                print("***")
                if len(good) > MIN_MATCH_COUNT:
                    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                result, (center1, center2) = self.warpImages(img2, img1, M)

                if shift > 0:
                    angle = self.get_2d_course_angle_image_coord_sys(x1 = self.points[shift][0][0], y1 = self.points[shift][0][1],x2 = self.points[shift - 1][0][0], y2 = self.points[shift - 1][0][1])
                    print("angle: ", angle)
                    distance = self.calculate_distance_xy(x1 = self.points[shift][0][0], y1 = self.points[shift][0][1],x2 = self.points[shift - 1][0][0], y2 = self.points[shift - 1][0][1])
                    print("distanse: ", distance)
                    
                    angles.append(angle)
                    distances.append(distance)

                    print(">>> ", distance)
                    print(">>> ", angle)



                # Store the center coordinates
                centers.append(center1)  # Center of img1 in result coordinates
                centers.append(center2)  # Center of img2 in result coordinates

                if result.shape[0] > self.size_edje or result.shape[1] > self.size_edje:
                    now = str(datetime.now()).replace(" ","_").replace(":","_").replace("-","_")
                    transform, img = self.__add_alpha_chanel(img=result)
                    image_path = str("./res/") + str(now) + self.output_image
                    self.__tiff_saving(img=img, transform=transform, path=image_path)

                    del img_list[0]
                    img_list = self.__insert_image_buffer(img_list=img_list)
                    result = img_list[0]
                    shift += 1

                img_list.insert(0, result)
                shift += 1

            except Exception as ex:
                print("[warning] Stopped. Exeption")
                print(ex)
                now = str(datetime.now()).replace(" ","_").replace(":","_").replace("-","_")
                transform, img = self.__add_alpha_chanel(img=result)
                image_path = str("./res/") + str(now) + self.output_image
                self.__tiff_saving(img=img, transform=transform, path=image_path)

                del img_list[0]
                img_list.insert(0, self.image_list_buffer[0])
                img_list.insert(1, self.image_list_buffer[1])
                result = img_list[0]
                shift += 1
                break

            if len(img_list) <3:
                print("[warning] Stopped. All images in use")
                now = str(datetime.now()).replace(" ","_").replace(":","_").replace("-","_")
                transform, img = self.__add_alpha_chanel(img=result)
                image_path = str("./res/") + str(now) + self.output_image
                self.__tiff_saving(img=img, transform=transform, path=image_path)
                break

        print("Centers of images in stitched result:", centers)  # Log the centers
        return result, centers, angles, distances, img, transform, image_path

    def draw_all_centers(self, angles, distances, img, transform, image_path):
        print(">>> ", angles)
        print(">>> ", distances)
        last_point = self.points[len(self.points) - 1][0]
        distances.reverse()
        angles.reverse()

        print("Last point: ", last_point)
        print("Distances: ", distances)
        print("Angles: ", angles)

        print("distances_rev: ", distances)
        print("angles_rev: ", angles)

        for i in range(len(self.points) - 1):
            try:
                print("*")
                x, y = self.get_xy_by_course_and_distance(x = last_point[0], y = last_point[1], 
                    l = distances[i], azimuth = angles[i])
                print("x,y: ", x,y)
                cv2.line(img, tuple(last_point), (int(x), int(y)), (0, 255, 255), 10) 
                last_point = (int(x), int(y))
            except:
                print(":c")
        self.__tiff_saving(img=img, transform=transform, path=image_path)
        return 0



    def main(self):
        file_list = self.__read_files(path = self.image_path)
        print(file_list)
        img_list = self.__create_image_list(lst = file_list)
        result, centers, angles, distances, img, transform, image_path = self.stiching(img_list = img_list, file_list = file_list)
        result = self.draw_all_centers(angles = angles, distances = distances, img = img, transform = transform, image_path = image_path)
        


if __name__ == "__main__":
    pm = Panoranic_mode(descriptor_algo = "orb", resize_coef = 0.8)
    pm.main()