#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import cv2
import numpy as np

import os
import argparse
import warnings
from datetime import datetime
from math import cos, sin, pi, pow, sqrt, radians, atan2, degrees, tan
from PIL import Image
import yaml
import torch
from tqdm import tqdm

import warnings



from exif_processing import Exif_Processing
from img_processing import Image_Processing


ap = argparse.ArgumentParser()
ap.add_argument("-img", "--input_image", default="./img18/", required = False,
    help = "Path to the directory that contains the imags")
ap.add_argument("-out", "--output_image", default="res.tif", required = False,
    help = "Path to the output image")
ap.add_argument("-set", "--path_default_settings", default="./config.yaml", required = False,
    help = "Path to config file")


args = vars(ap.parse_args())



import logging

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, 
            format='%(asctime)s - %(levelname)s - %(message)s', 
            datefmt='%Y-%m-%d %H:%M:%S')

class Panoranic_mode(object):
    def __init__(self, image_path = args["input_image"], verbose=True, 
        output_image = args["output_image"], settings = None):

        if settings is None:
            settings_path = args["path_default_settings"]
            with open(settings_path, "r") as file:
                self.settings = yaml.safe_load(file)
        else:
            self.settings = settings

        self.image_path = image_path
        self.output_image = output_image
        self.verbose = verbose

        # init parameters from config file
        self.max_num_keypoints = int(self.settings['descriptor']['max_num_keypoints'])
        self.device_type = str(self.settings['descriptor']['device'])
        self.trees = int(self.settings['descriptor']['trees'])
        self.k = int(self.settings['descriptor']['k'])
        self.dist_value = float(self.settings['descriptor']['dist_value'])
        self.resize_coef = float(self.settings['image']['resize_coef'])
        self.descriptor_algo = str(self.settings['descriptor']['algo'])
        self.nfeatures=int(self.settings['descriptor']['nfeatures'])
        self.size_edje = int(self.settings['image']['size_edje'])
        self.mode = str(self.settings['mode']['type'])
        self.step = int(self.settings['mode']['step'])
        self.chunk_size = int(self.settings['mode']['chunk_size'])
        self.overlap_size = int(self.settings['mode']['overlap_size'])


        # init choosen descriptor
        if self.descriptor_algo == "sift":
            self.descriptor_type = cv2.SIFT_create()  
        elif self.descriptor_algo == "orb":
            self.descriptor_type = cv2.ORB_create(nfeatures=self.nfeatures) 
        elif self.descriptor_algo =="brisk":
            self.descriptor_type = cv2.BRISK_create() 
        elif self.descriptor_algo =="lglue":
            self.device = torch.device("cuda" if torch.cuda.is_available() else self.device_type)

            if torch.cuda.is_available():
                logging.info(f"Cude is available")
            else:
                logging.warning(f"Cude is not available")


        

        self.image_list_buffer = [None, None, None]
        self.points = []
        self.img_buffer = None


    def __split_list(self, data, overlap_size = 2):
        # Разбивает список на подсписки заданного размера
        modern_list = [data[i:i + self.chunk_size] for i in range(0, len(data), self.chunk_size)]
        result = []
        for i in range(len(modern_list) - 1):
            combined = modern_list[i] + modern_list[i + 1][:overlap_size]
            result.append(combined)
        return result


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


    def __get_files_list(self, path):
        
        file_list = os.listdir(path)
        if self.mode == "video":
            file_list = file_list[::self.step]

        return sorted(file_list)


    def __create_image_list(self, lst):
        img_list = []
        for i in range(len(lst)):
            new_image = cv2.imread(self.image_path + lst[i])
            new_image = cv2.resize(new_image, (0, 0), fx = self.resize_coef, fy = self.resize_coef)
            img_list.append(new_image)
        return img_list





    def __calculate_distance_xy(self, delta_x, delta_y):
        """
        Method for dist calculation between 2 points in 2D
        input    -> p1, p2: (x1,y1), (x2,y2)
        output   -> dist 
        """
        if delta_x and delta_y :
            return sqrt(pow(abs(delta_x), 2) + pow(abs(delta_y),2))

        else:
            return None



    def __set_all_centers_coordinates(self, path, wgs_points):
        image_list = self.__get_files_list(path)
        print("[INFO] Setting coordinates ..")
        for i in tqdm(range(len(image_list))):
            image_path = self.image_path + image_list[i]
            lat, lon = wgs_points[i]
            Exif_Processing.set_gps_data(image_path, image_path, lat, lon)
        print("[INFO] Coordinates set!")
        return 0

    def set_descriptor(self, descriptors1, descriptors2):
        matches = None

        if self.descriptor_algo == "sift":
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees = self.trees)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(descriptors1, descriptors2, self.k)
                
        elif self.descriptor_algo == "orb":
            bf = cv2.BFMatcher_create(cv2.NORM_HAMMING)
            matches = bf.knnMatch(descriptors1, descriptors2, self.k)
                
        elif self.descriptor_algo == "brisk":
            bf = cv2.BFMatcher_create(cv2.NORM_HAMMING)
            matches = bf.knnMatch(descriptors1, descriptors2, self.k)

        return matches

    def draw_centers(self, centers, accumulated_image, file_list, center_offset, name = "new.jpg"):
        for idx, center in enumerate(centers):
            x, y = center
            x += center_offset[0]
            y += center_offset[1]
            if 0 <= int(x) < accumulated_image.shape[1] and 0 <= int(y) < accumulated_image.shape[0]:
                cv2.circle(accumulated_image, (int(x), int(y)), radius=10, color=(0, 0, 255), thickness=-1)

                cv2.putText(
                    accumulated_image,
                    f'{file_list[idx]}',
                    (int(x) + 15, int(y) - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )
                

        # Сохранение итогового изображения
        output_dir = "chunks"
        output_filename = str(datetime.now()).replace(" ","_").replace(":","_").replace("-","_")
        output_path = os.path.join(output_dir, output_filename)
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(output_path, accumulated_images)
             
                        



    def stiching(self, file_list=[], minmaxcount=5):
        print(" [INFO] Stiching.. ")
        centers = []  # List to store the center coordinates of all images
        homographies = []
        result = None
        transform = 0

        accumulated_image = None
        

        file_list_modern = self.__split_list(data = file_list, overlap_size = self.overlap_size)

        for file_list in file_list_modern:
            print("file_list: ", file_list)
            for g in tqdm(range(len(file_list))):
                if len(file_list) == 0:
                    try:
                        now = str(datetime.now()).replace(" ","_").replace(":","_").replace("-","_")
                        transform, img = Image_Processing().add_alpha_chanel(img=result)
                        image_path = str("./res/e") + str(now) + self.output_image
                        Image_Processing().tiff_saving(img=img, transform=transform, path=image_path)
                        print("[warning] Stopped. Not enought images")
                        break

                    except Exception as ex:
                        print(f"[warning] Not enought images. Some problem with image {ex}")
                        break

                try:

                    if  (g == 0):
                        img1_path = file_list.pop(0)
                        img1 = cv2.imread(self.image_path + img1_path)
                        img1 = cv2.resize(img1, (0, 0), fx = self.resize_coef, fy = self.resize_coef)
                        #print(img1_path, img1.shape)

                        # Центрируем первое изображение
                        base_img_height, base_img_width = img1.shape[:2]
                        base_center = (base_img_width / 2, base_img_height / 2)
                        centers.append(base_center)

                        # Создаём пустое изображение для накопленного результата
                        accumulated_image = np.zeros((20000, 20000, 3), dtype=np.uint8)
                        center_offset = (10000 - base_img_width // 2, 10000 - base_img_height // 2)
                        accumulated_image[
                            center_offset[1]:center_offset[1] + base_img_height,
                            center_offset[0]:center_offset[0] + base_img_width
                        ] = img1


                    elif(g > 0):
                        img1 = self.img_buffer
                        #print(" concated image: ", img1.shape)

                    img2_path = file_list.pop(0)
                    img2 = cv2.imread(self.image_path + img2_path)
                    img2 = cv2.resize(img2, (0, 0), fx = self.resize_coef, fy = self.resize_coef)


                    keypoints1, descriptors1 = self.descriptor_type.detectAndCompute(img1, None)
                    keypoints2, descriptors2 = self.descriptor_type.detectAndCompute(img2, None)

                    if descriptors1 is None or descriptors2 is None or len(keypoints1) == 0 or len(keypoints2) == 0:
                        print("[warning] Stopped. Not enought descriptors")
                        now = str(datetime.now()).replace(" ","_").replace(":","_").replace("-","_")
                        transform, result = Image_Processing().add_alpha_chanel(img=result)
                        image_path = str("./res/d") + str(now) + self.output_image
                        Image_Processing().tiff_saving(img=result, transform=transform, path=image_path)
                        self.img_buffer = img2
                        
                    

                    matches = self.set_descriptor(descriptors1, descriptors2)                


                    good = [m for m, n in matches if m.distance < self.dist_value * n.distance]
                    
                    print("Good: ",len(good))
                    MIN_MATCH_COUNT = minmaxcount
                
                    result = None

                    if len(good) > MIN_MATCH_COUNT:
                        
                        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

                        #src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good[:3]])  
                        #dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good[:3]])
                        
                        
                        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                        homographies.append(H)

                        curr_img_height, curr_img_width = img2.shape[:2]
                        curr_center = (curr_img_width / 2, curr_img_height / 2)

                        result = Image_Processing().warpImages(img2, img1, H)
                        



                        '''
                        H = cv2.getAffineTransform(src_pts, dst_pts)
                        homographies.append(H)
                        curr_img_height, curr_img_width = img2.shape[:2]
                        curr_center = (curr_img_width / 2, curr_img_height / 2)
                        result = cv2.warpAffine(img2, H, (curr_img_width, curr_img_height))
                        '''




    
                        result, x, y, w, h = Image_Processing().cut_black_part(img = result)
                    

                        self.img_buffer = result

                    else:
                        #transformation_matrix.append(0)
                        print("[WARING] No enought matching!!! ")
                        if result is not None:
                            now = str(datetime.now()).replace(" ","_").replace(":","_").replace("-","_")
                            transform, result = Image_Processing().add_alpha_chanel(img=result)
                            image_path = str("./res/c") + str(now) + self.output_image
                            Image_Processing().tiff_saving(img=result, transform=transform, path=image_path)
                            self.img_buffer = img2
                        print("[WARNING] Image is None!")



                    if result is not None:
                        if result.shape[0] > self.size_edje or result.shape[1] > self.size_edje:
                            now = str(datetime.now()).replace(" ","_").replace(":","_").replace("-","_")
                            transform, result = Image_Processing().add_alpha_chanel(img=result)
                            image_path = str("./res/b") + str(now) + self.output_image
                            Image_Processing().tiff_saving(img=result, transform=transform, path=image_path)
                            self.img_buffer = img2

                        


                

                except cv2.error as e:
                    if "cv::OutOfMemoryError" in str(e):
                        print("Caught OpenCV OutOfMemoryError: Reducing image size or freeing memory.")
                        if result is not None:
                            now = str(datetime.now()).replace(" ","_").replace(":","_").replace("-","_")
                            transform, result = Image_Processing().add_alpha_chanel(img=result)
                            image_path = str("./res/a") + str(now) + self.output_image
                            Image_Processing().tiff_saving(img=result, transform=transform, path=image_path)
                            self.img_buffer = img2
                        print(" >>> save")


                    else:
                        print(">>> raise")
                        raise  # Re-raise if it's a different OpenCV error


                except MemoryError:
                    print("Caught Python MemoryError: consider reducing image resolution.")


                except Exception as ex:
                    print(f"[warning] Stopped. {ex}")
                    break


        return 0

    
    '''
    def draw_all_centers(self, transformation_matrix, img, draw_flag = True):
        for i in range(len(transformation_matrix)):
            params = Image_Processing().extract_rotation_translation(transformation_matrix[i])
            #print(f"{params}")
            #print("Dist: ", self.__calculate_distance_xy(delta_x = params['translation'][0], delta_y = params['translation'][1]))
            rotation_deg = (self.__rad_to_degree(radians = params['rotation_degrees']) + 180) % 360 - 180
            #rotation_deg = self.__rad_to_degree(radians = params['rotation_degrees'])
            
            #print("Angle: ", rotation_deg)
            #print("----------------")
            #if draw_flag is True:
            #s    cv2.circle(img, (pixel_x, pixel_y), radius, color, -1)

        return 0

        last_point = self.points[len(self.points) - 1][0]


        print("Last point: ", last_point)
    


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
        self.tiff_saving(img=img, transform=transform, path=image_path)
        '''





    def __get_coordinates_list(self, path = None):
        if path is None:
            path = self.image_path

        files_list = self.__get_files_list(path = path)
        coordinates = []
        print("[INFO] Get all gps coordinates from exif ..")

        for i in tqdm(range(len(files_list))):
            image_path = self.image_path + files_list[i]
            latitude, longitude = self.get_gps(image_path)

            coordinates.append((longitude, latitude))
            #if latitude and longitude:
            #    print(f"Lat = {latitude}, Lon = {longitude}")
            #else:
            #    print("No GPS data found.")
        return coordinates





    def main(self):
        file_list_sorted = self.__get_files_list(path = self.image_path) #overlap_size = self.overlap_size
        print(f"[INFO] Detected {len(file_list_sorted)} images in the folder: {self.image_path} ")
        
        #file_list_sorted = file_list_sorted[::30]
        #print(f"[INFO] In use: {len(file_list_sorted)} ")

        #img_list = self.__create_image_list(lst = file_list)
        res = self.stiching(file_list = file_list_sorted)
        print(" [INFO] Final")
        #result = self.draw_all_centers(transformation_matrix = transformation_matrix, img = result)
        '''
        self.debug_draw_gps_centers_on_orto()
        '''

        '''
        wgs_points = self.__get_all_centers_coordinates(geotiff_path = './odm_orthophoto.tif', points = [])
        reult = self.__set_all_centers_coordinates(wgs_points = wgs_points, path = self.image_path)
        '''
        


if __name__ == "__main__":
    settings = None

    with open("config.yaml", "r") as file:
        settings = yaml.safe_load(file)

    pm = Panoranic_mode(settings = settings)
    pm.main()
    
