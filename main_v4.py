#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import cv2
import numpy as np

import os
import glob
import argparse
import warnings

from PIL import Image
import yaml

from tqdm import tqdm

import warnings
import logging


from exif_processing import Exif_Processing
import img_processing





ap = argparse.ArgumentParser()
ap.add_argument("-img", "--input_image", default="./img8/", required = False,
    help = "Path to the directory that contains the imags")
ap.add_argument("-out", "--output_image", default="./chunks/", required = False,
    help = "Path to the output image")
ap.add_argument("-ortho", "--ortho_referenced_path", default="./src/4326.tif", required = False,
    help = "Path to the geo referenced orthophoto ")
ap.add_argument("-set", "--path_default_settings", default="./config.yaml", required = False,
    help = "Path to config file")



args = vars(ap.parse_args())




warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, 
            format='%(asctime)s - %(levelname)s - %(message)s', 
            datefmt='%Y-%m-%d %H:%M:%S')


class Panoranic_mode(object):
    def __init__(self, image_path = args["input_image"], verbose=True, 
        output_image = args["output_image"], ortho_referenced_path = args["ortho_referenced_path"], 
        settings = None):

        if settings is None:
            settings_path = args["path_default_settings"]
            with open(settings_path, "r") as file:
                self.settings = yaml.safe_load(file)
        else:
            self.settings = settings

        self.image_path = image_path
        self.output_image = output_image
        self.ortho_referenced_path = ortho_referenced_path
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
        self.chunk_ortho_side_size = int(self.settings['mode']['chunk_ortho_side_size'])
        self.good_matches_quantity = int(self.settings['descriptor']['good_matches_quantity'])
        self.min_match_count = int(self.settings['descriptor']['min_match_count'])

        # init choosen descriptor
        if self.descriptor_algo == "sift":
            self.descriptor_type = cv2.SIFT_create()
            print("sift")  
        elif self.descriptor_algo == "orb":
            self.descriptor_type = cv2.ORB_create(nfeatures=self.nfeatures) 
        elif self.descriptor_algo =="brisk":
            self.descriptor_type = cv2.BRISK_create() 

        self.img_proc = img_processing.Image_Processing(max_num_keypoints = self.min_match_count, verbose = True, device_type = self.device_type, output_image = self.output_image)
        


    def __split_list(self, data, overlap_size = 2):
        # Разбивает список на подсписки заданного размера
        modern_list = [data[i:i + self.chunk_size] for i in range(0, len(data), self.chunk_size)]
        result = []
        for i in range(len(modern_list) - 1):
            combined = modern_list[i] + modern_list[i + 1][:overlap_size]
            result.append(combined)
        return result



    def __get_images_paths_list(self, directory, extensions=None):

        if extensions is None:
            extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.tiff']

        file_list = []
        for ext in extensions:
            file_list.extend(glob.glob(os.path.join(directory, ext)))

        if self.mode == "video":
            file_list = file_list[::self.step]

        return sorted(file_list)


    def __create_image_list(self, lst):
        img_list = []
        for i in range(len(lst)):
            new_image = cv2.imread(str(lst[i]))
            new_image = cv2.resize(new_image, (0, 0), fx = self.resize_coef, fy = self.resize_coef)
            img_list.append(new_image)
        return img_list


    def __set_all_centers_coordinates(self, path, wgs_points):
        image_list = self.__get_images_paths_list(directory = path)
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

             
                        
    def create_empty_image_with_first_image(self, images):
        centers = []
        # Центрируем первое изображение
        base_img_height, base_img_width = images[0].shape[:2]
        base_center = (base_img_width / 2, base_img_height / 2)
        centers.append(base_center)

        # Создаём пустое изображение для накопленного результата
        accumulated_image = np.zeros((self.chunk_ortho_side_size, self.chunk_ortho_side_size, 3), dtype=np.uint8)
        center_offset = (int(self.chunk_ortho_side_size / 2 - base_img_width // 2), int(self.chunk_ortho_side_size / 2 - base_img_height // 2))
        accumulated_image[
            center_offset[1]:center_offset[1] + base_img_height,
            center_offset[0]:center_offset[0] + base_img_width
        ] = images[0]

        return accumulated_image, center_offset, centers


    def stiching(self, file_list=[]):
        
        centers_gcps_db = []

        file_list_modern = self.__split_list(file_list)

        print(f"There are: {len(file_list_modern)} chunks")
        print(f" [INFO] Stiching.. ")

        for file_list_iter in file_list_modern:
            images = self.__create_image_list(lst = file_list_iter)
        
            # Параметры для хранения центров и гомографий
            homographies = []

            accumulated_image, center_offset, centers = self.create_empty_image_with_first_image(images)

            centers_gcps = {}

            # Итерация по парам изображений для вычисления гомографий и центров
            for i in tqdm(range(1, len(images))):
                keypoints_prev, descriptors_prev = self.descriptor_type.detectAndCompute(images[i - 1], None)
                keypoints_curr, descriptors_curr = self.descriptor_type.detectAndCompute(images[i], None)
                    
                
                matches = self.set_descriptor(descriptors_prev, descriptors_curr)
                good = [m for m, n in matches if m.distance < self.dist_value * n.distance]


                if len(matches) < 10:
                    raise ValueError(f"Not enough matches found between image {i} and image {i-1}.")
                #print("matches: ", len(matches))
                #print("good: ", len(good))
                if len(good) > self.min_match_count:
                    src_pts = np.float32([keypoints_prev[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                    dst_pts = np.float32([keypoints_curr[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

                    H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
                    homographies.append(H)

                    curr_img_height, curr_img_width = images[i].shape[:2]
                    curr_center = (curr_img_width / 2, curr_img_height / 2)
                    transformed_center = np.array([[curr_center[0], curr_center[1]]], dtype="float32").reshape(-1, 1, 2)

                    for j in range(i):
                        transformed_center = cv2.perspectiveTransform(transformed_center, homographies[j])

                    centers.append((transformed_center[0][0][0], transformed_center[0][0][1]))

                    offset_H = np.eye(3)
                    offset_H[0, 2] = center_offset[0]
                    offset_H[1, 2] = center_offset[1]
           
                    centers_gcps[file_list_iter[i]]=(int(transformed_center[0][0][0] + center_offset[0]), int(transformed_center[0][0][1]+ center_offset[1])) #(x,y)

                    if len(homographies[:i]) > 1:
                        full_H = offset_H @ np.linalg.multi_dot(homographies[:i][::-1])
                    else:
                        full_H = offset_H @ homographies[0]

                    warped_image = cv2.warpPerspective(images[i], full_H, (accumulated_image.shape[1], accumulated_image.shape[0])) #, flags=cv2.INTER_LINEAR
                    mask = (warped_image > 0)
                    accumulated_image[mask] = warped_image[mask]
                    
            #print(centers_gcps)

            centers_gcps_db.append(centers_gcps)

            self.img_proc.draw_centers(centers, accumulated_image, file_list_iter, center_offset)
                        
            self.img_proc.jpg_saving(img = accumulated_image)


        return centers_gcps_db
        
    
    
    
    def search_projectoins_positions(self, device_type): #centers_gcps_db
        print(f" [INFO] Searching projections.. ")

        # scaling
        scaled_ortho = self.img_proc.resize_geotiff(input_path = self.ortho_referenced_path, output_path = './src/4326_1.tif', scale_factor = 0.5)
        print(" [INFO] Size of the scaled ortho: ", scaled_ortho.shape[2], scaled_ortho.shape[1])

        # read scaled ortho
        wgs_points = []
        lons_big, lats_big = self.img_proc.pix2coord(path = './src/4326_1.tif') 
        
        # read geo referenced ortho
        ortho = self.img_proc.load_picture(image_path = './src/4326_1.tif')
        ortho_file = self.img_proc.read_tiff_file(path = './src/4326_1.tif')


        ortho = self.img_proc.tiff_2_ndarray(ortho_file)
        print(" [INFO] Size of the ortho: ", ortho_file.width, ortho_file.height)
        
        #ortho = self.img_proc.rasterio_to_cv2format(img = ortho)
        #print("ortho: ", ortho.shape)

        #img1 = self.img_proc.load_tiff_image('./src/4326_1.tif') 
        #print("tiff shape", img1.shape)
        #img1_gray = self.img_proc.convert_to_grayscale(ortho)

        chunk_list = self.__get_images_paths_list(directory = self.output_image)
      
        # iterative read of each chunk
        for i in range(len(chunk_list)):

            print("path: ", str(chunk_list[i]))

            path = chunk_list[i].replace('\\','/')
            #chunk = cv2.imread(path)
            #chunk = self.img_proc.cv2_to_tensor(image = chunk)
            chunk = self.img_proc.load_jpg_image(path)  # JPG image
            print("chunk shape", chunk.shape)
            img2_gray =  self.img_proc.convert_to_grayscale(chunk)


            
            print("Chunk type: ", type(chunk))
            print("chunk: ", chunk.shape)

            '''if chunk is None or not isinstance(chunk, np.ndarray):
                raise ValueError("Invalid image data. Ensure 'chunk' is a properly loaded image as a NumPy array.")

            # Convert to grayscale if it's a color image
            if len(chunk.shape) == 3:
                chunk = cv2.cvtColor(chunk, cv2.COLOR_BGR2GRAY)
            
            keypoints_curr, descriptors_curr = self.descriptor_type.detectAndCompute(chunk, None)
            keypoints_prev, descriptors_prev = self.descriptor_type.detectAndCompute(ortho, None)
                    
            
            matches = self.set_descriptor(descriptors_prev, descriptors_curr)
            good = [m for m, n in matches if m.distance < self.dist_value * n.distance]'''



             
            

            # Convert to grayscale (if not already)
            
            
            print(ortho.shape)
            print(chunk.shape)
            # Compute SIFT descriptors
            kp1, des1, kp2, des2 =  self.img_proc.compute_sift_descriptors(ortho, chunk)

            # Match descriptors
            good_matches =  self.img_proc.match_descriptors(des1, des2)

           



            for j in tqdm(range(len(m_kpts0))):

                # getting coordiates in WGS84 CS from each pair from both orthos
                lat_big = round(lats_big[m_kpts0[j][1].numpy().astype(int)][m_kpts0[j][0].numpy().astype(int)] / 100000, 6)
                lon_big = round(lons_big[m_kpts0[j][1].numpy().astype(int)][m_kpts0[j][0].numpy().astype(int)] / 10000, 6)
                #lat_small = round(lats_small[m_kpts1[i][1].numpy().astype(int)][m_kpts1[i][0].numpy().astype(int)] / 100000,6)
                #lon_small = round(lons_small[m_kpts1[i][1].numpy().astype(int)][m_kpts1[i][0].numpy().astype(int)] / 10000,6)
                print(lat_big, lon_big)

            #H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)


            #transformed_center = np.array([[curr_center[0], curr_center[1]]], dtype="float32").reshape(-1, 1, 2)
                                                                                                                                                                          

            #transformed_center = cv2.perspectiveTransform(transformed_center, H)


        return wgs_points



    

    def main(self):
        file_list_sorted = self.__get_images_paths_list(directory = self.image_path)
        #print(f"[INFO] Detected {len(file_list_sorted)} images in the folder: {self.image_path} ")
        
        centers_gcps_db = self.stiching(file_list = file_list_sorted)
        
        wgs_points = self.search_projectoins_positions(device_type = self.device_type) #centers_gcps_db = centers_gcps_db
        #print(wgs_points)
        


        '''
        self.debug_draw_gps_centers_on_orto()
        '''

        
        wgs_points = self.__get_all_centers_coordinates(geotiff_path = './odm_orthophoto.tif', points = [])
        reult = self.__set_all_centers_coordinates(wgs_points = wgs_points, path = self.image_path)
        
        


if __name__ == "__main__":
    settings = None

    with open("config.yaml", "r") as file:
        settings = yaml.safe_load(file)

    pm = Panoranic_mode(settings = settings)
    pm.main()
    
