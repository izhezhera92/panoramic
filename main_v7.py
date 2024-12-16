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
import shutil
import warnings
import logging


from exif_processing import Exif_Processing
from img_processing import Image_Processing
from scenario import Scenario
import files_utils 
import user_utils


ap = argparse.ArgumentParser()
ap.add_argument("-img", "--input_image", default="E:/farsightvision/67/test/", required = False,
    help = "Path to the directory that contains the imags")
ap.add_argument("-out", "--output_image", default="res.tif", required = False,
    help = "Path to the output image")
ap.add_argument("-set", "--path_default_settings", default="./config.yaml", required = False,
    help = "Path to config file")
ap.add_argument("-temp_a", "--path_temp_a", default="./temp_a/", required = False,
    help = "Path to temp files")
ap.add_argument("-temp_b", "--path_temp_b", default="./temp_b/", required = False,
    help = "Path to temp files")


args = vars(ap.parse_args())


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
        self.max_num_keypoints = int(self.settings['descriptor']['max_num_keypoints'])
        self.device_type = str(self.settings['descriptor']['device'])
        self.resize_coef = float(self.settings['image']['resize_coef'])
        self.nfeatures=int(self.settings['descriptor']['nfeatures'])
        self.size_edje = int(self.settings['image']['size_edje'])
        self.mode = str(self.settings['mode']['type'])
        self.step = int(self.settings['mode']['step'])
        self.chunk_size = int(self.settings['mode']['chunk_size'])
        self.overlap_size = int(self.settings['mode']['overlap_size'])
        self.start = int(self.settings['mode']['start'])
        self.last = int(self.settings['mode']['last'])
        self.part_or_full = bool(self.settings['mode']['part_or_full'])
        self.reverse = bool(self.settings['mode']['reverse'])
        self.ortho_type = str(self.settings['mode']['ortho_type'])
        self.temp_path_a = args["path_temp_a"]
        self.temp_path_b = args["path_temp_b"]
        self.descriptor_algo = str(self.settings['descriptor']['algo'])
        self.img_buffer = None
        self.impr = Image_Processing(settings = settings)
        self.iter_value = float(self.settings['descriptor']['iter_value'])
                        


    def stiching(self, file_list=[], minmaxcount=5):
        logging.info(f"Stiching..")
        start_time = datetime.now()
        homographies = []
        result = None
        transform = 0        

        #file_list = files_utils.modificat_list(data = file_list)

        if self.ortho_type == "way":
            file_list_modern = files_utils.split_list(data = file_list, chunk_size = self.chunk_size, overlap_size = self.overlap_size)


        for file_list in file_list_modern:
            print("file_list: ", file_list)
            homographies = []
            file_list_accum = []
            for g in range(len(file_list)):

                if len(file_list) == 0:
                    try:
                        self.impr.chunk_saving(result = result, path = self.temp_path_a, homographies = homographies, file_list_accum = file_list_accum)
                        logging.warning(f"Stiching stopped. Buffer of images is empty.")
                        break

                    except Exception as ex:
                        logging.error(ex)
                        break

                try:
                    if  (g == 0):
                        img1_path = file_list.pop(0)
                        img1 = cv2.imread(self.image_path + img1_path)
                        img1 = cv2.resize(img1, (0, 0), fx = self.resize_coef, fy = self.resize_coef)
                        file_list_accum.append(img1_path)


                    elif(g > 0):
                        img1 = self.img_buffer


                    img2_path = file_list.pop(0)
                    img2 = cv2.imread(self.image_path + img2_path)
                    img2 = cv2.resize(img2, (0, 0), fx = self.resize_coef, fy = self.resize_coef)
                    file_list_accum.append(img2_path)

                    #img1 = self.impr.adjust_gamma(img = img1)
                    #img2 = self.impr.adjust_gamma(img = img2)

                    if img1.shape[2] == 4:  # Если изображение имеет альфа-канал
                        img1 = cv2.cvtColor(img1, cv2.COLOR_RGBA2RGB)

                    if img2.shape[2] == 4:  # Аналогично для второго изображения
                        img2 = cv2.cvtColor(img2, cv2.COLOR_RGBA2RGB)

                 
                    keypoints1, descriptors1, keypoints2, descriptors2 = self.impr.get_keyPoints_and_descriptors(img1, img2)

                    if descriptors1 is None or descriptors2 is None or len(keypoints1) == 0 or len(keypoints2) == 0:
                        logging.waring("Stiching stopped. Not enought descriptors.")
                        self.impr.chunk_saving(result = result, path = self.temp_path_a, homographies = homographies, file_list_accum = file_list_accum)
                        self.img_buffer = img2
                        
                        if descriptors1 is None or len(descriptors1) == 0:
                            logging.waring("Descriptors array descriptors1 is empty")

                        if descriptors2 is None or len(descriptors2) == 0:
                            logging.waring("Descriptors array descriptors2 is empty")


                    matches = self.impr.set_descriptor(descriptors1 = descriptors1, descriptors2 = descriptors2)
                    good = self.impr.matches_filtering(matches = matches)
                    
                    result = None

                   
                    if len(good) > minmaxcount:
                        result, x, y, w, h, homographies = self.impr.core_stiching(good = good, 
                            keypoints1 = keypoints1, keypoints2 = keypoints2, homographies = homographies, img2 = img2, img1 = img1, iter_value = self.iter_value)    
                    
                        self.img_buffer = result

                    else:
                        logging.waring(f"No enought matching! ({len(good)} pairs)")
                        if result is not None:
                            self.impr.chunk_saving(result = result, path = self.temp_path_a, homographies = homographies, file_list_accum = file_list_accum)
                            self.img_buffer = img2
                        logging.waring(f"Image is None!")


                    if result is not None:
                        if result.shape[0] > self.size_edje or result.shape[1] > self.size_edje:
                            self.impr.chunk_saving(result = result, path = self.temp_path_a, homographies = homographies, file_list_accum = file_list_accum)
                            self.img_buffer = img2
                

                except cv2.error as e:
                    if "cv::OutOfMemoryError" in str(e):
                        logging.error(f"Caught OpenCV OutOfMemoryError: Reducing image size or freeing memory.")
                        resize_coef_temp = pow(self.resize_coef, 2)
                        img1 = cv2.resize(img1, (0, 0), fx = resize_coef_temp, fy = resize_coef_temp)
                        
                        keypoints1, descriptors1, keypoints2, descriptors2 = self.impr.get_keyPoints_and_descriptors(img1, img2)

                        if descriptors1 is None or descriptors2 is None or len(keypoints1) == 0 or len(keypoints2) == 0:
                            plogging.waring("Stiching stopped. Not enought descriptors.")

                            self.impr.chunk_saving(result = result, path = self.temp_path_a, homographies = homographies)
                            self.img_buffer = img2
                            
                        
                        matches = self.impr.set_descriptor(algo = self.descriptor_algo, descriptors1 = descriptors1, descriptors2 = descriptors2)
                        good = self.impr.matches_filtering(matches = matches)
                    
                        result = None

                        if len(good) > minmaxcount:
                            
                            result, x, y, w, h, homographies = self.impr.core_stiching(good = good, 
                                keypoints1 = keypoints1, keypoints2 = keypoints2, homographies = homographies, img2 = img2, img1 = img1, iter_value = self.iter_value)
                    
                            self.img_buffer = result
                            logging.info(f"Resizing issue had solved")


                    else:
                        logging.error(e)
                        raise  
                    


                except MemoryError:
                    logging.error(f"Caught Python MemoryError: consider reducing image resolution.")
                    if result is not None:
                        self.impr.chunk_saving(result = result, path = self.temp_path_a, homographies = homographies, file_list_accum = file_list_accum)
                        self.img_buffer = img2


                except Exception as ex:
                    logging.error(ex)
                    break


        files_utils.move_files_from_a_to_b(path_a = self.temp_path_a, path_b = self.temp_path_b)
        user_utils.send_report_to_email(message = "Stiching process had finished", time_start = str(start_time), image = datetime.now())

        return 0




    def main(self):
        files_utils.buffer_folder_preparation(temp_path_a = self.temp_path_a, temp_path_b = self.temp_path_b)
        #file_list_sorted = self.get_files_list(path = self.image_path) #overlap_size = self.overlap_size
        file_list_sorted = files_utils.get_files_list(path = self.image_path, mode = self.mode, step = self.step, reverse = self.reverse, part_or_full = self.part_or_full, start = self.start, last = self.last)
        logging.info(f"Detected {len(file_list_sorted)} images in the folder: {self.image_path}")
        

        #file_list_sorted = file_list_sorted[::30]
        #print(f"[INFO] In use: {len(file_list_sorted)} ")

        #img_list = files_utils.create_image_list(lst = file_list, image_path = self.image_path, resize_coef = self.resize_coef)
        res = self.stiching(file_list = file_list_sorted)
        logging.info(" [INFO] Final")
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
    






