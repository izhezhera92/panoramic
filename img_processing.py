import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

import rasterio
from rasterio.transform import from_origin
from rasterio.warp import transform
from rasterio.enums import Resampling
from rasterio.plot import reshape_as_image

from lightglue import LightGlue, SuperPoint, DISK, viz2d
from lightglue.utils import load_image, rbd

import torch

class Image_Processing(object):
    def __init__(self, max_num_keypoints: int = 2048, 
        verbose: bool = True, 
        device_type: str = "cpu", 
        output_image: str = "./chunks/"):

        self.max_num_keypoints = max_num_keypoints
        self.verbose = verbose
        self.device_type = device_type
        self.output_image = output_image

        self.device = torch.device("cuda" if torch.cuda.is_available() else self.device_type)

        if torch.cuda.is_available():
            #logging.info(f"Cude is available")
            print(f" [INFO] CUDA is available")
        else:
            #logging.warning(f"Cude is not available")
            print(f" [INFO] CUDA is not available")


    def load_picture(self, image_path):
        return load_image(image_path)

    def read_tiff_file(self, path):
        return rasterio.open(path)
        

    def pix2coord(self, path: str = '', verbose: bool = True)-> list:
        ''' Method for geo data importing from geotif'''
        if path is not None:
            with rasterio.open(path) as src:
                band1 = src.read(1)
                #if self.verbose:
                #    logging.info(f"{path} has shape: {band1.shape}")
                height = band1.shape[0]
                width = band1.shape[1]

                # return a tuple of coordinate matrices from coordinate vectors
                cols, rows = np.meshgrid(np.arange(width), np.arange(height))
                xs, ys = rasterio.transform.xy(src.transform, rows, cols)
                lons = np.array(xs)
                lats = np.array(ys)
            return lons, lats
        return [], []

     
            

    def pixel_to_wgs84(self, img, points):
        wgs_points = []

        for i in range(len(points)):
            print(">>> ", points[i][0], points[i][1])
            # Get the pixel's geographic coordinates (in the img's CRS)
            point = img.transform * (points[i][0], points[i][1])
            src_crs = img.crs
            lon, lat = transform(src_crs, 'EPSG:4326', [point[0]], [point[1]])
            wgs_points.append([lat[0], lon[0]])

        return wgs_points


    def __get_all_centers_coordinates(self, geotiff_path, points = []):
        if points:
            with rasterio.open(geotiff_path) as img:
                return self.pixel_to_wgs84(img = img, points = points)

        return None

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
        warped_img2 = cv2.warpPerspective(img2, H_translation.dot(H), (x_max - x_min, y_max - y_min), flags=cv2.INTER_LINEAR)

        # Create a blank output image with the same size as the result
        output_img = np.zeros((y_max - y_min, x_max - x_min, 3), dtype=np.float32)

        # Paste the first image into the stitched result
        translated_img1 = np.zeros_like(output_img, dtype=np.float32)
        translated_img1[translation_dist[1]:rows1 + translation_dist[1], translation_dist[0]:cols1 + translation_dist[0], :] = img1.astype(np.float32)

        # Create masks for the two images
        mask1 = (translated_img1 > 0).astype(np.float32)
        mask2 = (warped_img2 > 0).astype(np.float32)

        # Calculate weighted blending in overlapping regions
        combined_mask = mask1 + mask2
        combined_mask[combined_mask == 0] = 1  # Avoid division by zero

        # Blend the images
        output_img = (translated_img1 + warped_img2) / combined_mask

        # Convert back to uint8 for display or saving
        output_img = np.clip(output_img, 0, 255).astype(np.uint8)

        return output_img
    '''
    # warping without transporent

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
        output_img = cv2.warpPerspective(img2, H_translation.dot(H), (x_max - x_min, y_max - y_min), flags=cv2.INTER_LINEAR) 

        # Paste the first image into the stitched result
        output_img[translation_dist[1]:rows1 + translation_dist[1], translation_dist[0]:cols1 + translation_dist[0], :3] = img1


        # Return the result image along with the centers
        return output_img
    '''


    def __rad_to_degree(self, radians):
        return np.degrees(radians)


    def debug_draw_gps_centers_on_orto(self, geotiff_path = "./ort.tif"):
        coordinates_list = self.__get_coordinates_list() 
        files_list = self.__get_files_list(path = self.image_path)
        print(coordinates_list)

        with rasterio.open(geotiff_path) as dataset:
            image_data = dataset.read(1)  
            
            pixel_points = [dataset.index(lon, lat) for lon, lat in coordinates_list]

        
        plt.figure(figsize=(10, 10))
        plt.imshow(image_data, cmap='gray')  
        i = 0
        
        for (x, y) in pixel_points:
            plt.plot(y, x,marker='o', markersize=5, color='red')
            info = str(i) + str(" ") + str(files_list[i])
            plt.text(y + 5, x + 5, info, color='blue', fontsize=7)
            i = i + 1

        plt.axis('off') 
        plt.show()


    def cut_black_part(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            x, y, w, h = cv2.boundingRect(contours[0])
            cropped_image = img[y:y+h, x:x+w]
            return cropped_image, x, y, w, h
        return 0, 0, 0, 0, 0


    def get_center_index(self, num_images):
        return num_images // 2


    def extract_rotation_translation(self, M):
        # Translation
        tx, ty = M[0, 2], M[1, 2]

        # Rotation angle (in radians) assuming affine transform
        rotation_rad = np.arctan2(M[1, 0], M[0, 0])
        rotation_deg = self.__rad_to_degree(rotation_rad)  # Convert to degrees

        # Optional: You can check the scale factors if needed (though not always reliable with homography)
        scale_x = np.sqrt(M[0, 0]**2 + M[1, 0]**2)
        scale_y = np.sqrt(M[0, 1]**2 + M[1, 1]**2)

        return {
            'translation': (tx, ty),
            'rotation_degrees': rotation_deg,
            'scale_x': scale_x,
            'scale_y': scale_y
        }


    def draw_centers(self, centers, accumulated_image,file_list_iter, center_offset):
        for idx, center in enumerate(centers):
            x, y = center
            x += center_offset[0]
            y += center_offset[1]

            if 0 <= int(x) < accumulated_image.shape[1] and 0 <= int(y) < accumulated_image.shape[0]:
                cv2.circle(accumulated_image, (int(x), int(y)), radius=10, color=(0, 0, 255), thickness=-1)

                cv2.putText(
                    accumulated_image,
                    f'{file_list_iter[idx]}',
                    (int(x) + 15, int(y) - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )
        return 0


    def tiff_saving(self, img, transform, path = './output_image.tif'):
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


    def jpg_saving(self, img, path = None):
        if path is None:
            path = self.output_image
        os.makedirs(path, exist_ok=True)
        now = str(datetime.now()).replace(" ","_").replace(":","_").replace("-","_")
        image_path = path + str(now) + '.jpg'
        res = cv2.imwrite(image_path, img)
        return 0


    def add_alpha_chanel(self, img):
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


    def rasterio_to_cv2format(self, img):
        # Print shape of the image for debugging
        print("Original image shape:", img.shape)

        # Check if the image has multiple bands (3D array) and has more than one band
        if len(img.shape) == 3 and img.shape[0] > 1:
            # If it's a multi-band image, move the bands axis (0th axis) to the last axis
            img = np.moveaxis(img, 0, -1)
            print("Image shape after moveaxis:", img.shape)
        
        # If the image is single-band (grayscale), no axis change is needed
        
        # Ensure data is in an OpenCV-compatible format (e.g., uint8)
        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        return img

    def cv2_to_tensor(self, image):
        """
        Convert an image from OpenCV (BGR) format to PyTorch tensor (RGB format).
        The image is normalized to the range [0, 1].
        """
        # Convert from BGR to RGB (OpenCV uses BGR by default)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert the image to a PyTorch tensor
        tensor = torch.from_numpy(image_rgb).float()

        # Normalize the image to [0, 1]
        tensor = tensor / 255.0

        # Rearrange the dimensions from (height, width, channels) to (channels, height, width)
        tensor = tensor.permute(2, 0, 1)  # Channels first (C, H, W)

        return tensor

    def matches_detection(self, image0, image1)-> tuple:
        ''' Method for the matches finding with LightGlue algorithm'''

        # setup SUperGlue detector
        print(f" [INFO] Matches detection.. ")
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


    def resize_geotiff(self, input_path, output_path, scale_factor): #input_path
        data = None
        with rasterio.open(input_path) as img:
            # Calculate the new dimensions
            new_width = int(img.shape[1] * scale_factor)
            new_height = int(img.shape[0] * scale_factor)

            # Update the transform to reflect the new dimensions
            transform = img.transform * img.transform.scale(
                (img.shape[1] / new_width),
                (img.shape[0] / new_height)
            )

            # Read and resample the data
            data = img.read(
                out_shape=(img.count, new_height, new_width),
                resampling=Resampling.bilinear  # or Resampling.nearest, depending on the data type
            )

            # Write the resized data to a new GeoTIFF
            with rasterio.open(
                output_path,
                'w',
                driver='GTiff',
                height=new_height,
                width=new_width,
                count=img.count,
                dtype=img.dtypes[0],
                crs=img.crs,
                transform=transform,
            ) as dst:
                dst.write(data)

        return data

    # Load a TIFF image using rasterio
    def load_tiff_image(self, file_path):
        with rasterio.open(file_path) as dataset:
            img_array = dataset.read(1)  # Read the first band (assuming grayscale)
            img = reshape_as_image(img_array)  # Reshape to 2D (height, width)
        return img

    def tiff_2_ndarray(self, dataset):
        img_array = dataset.read(1)
        return reshape_as_image(img_array)  # Reshape to 2D (height, width)


    # Load a JPG image using OpenCV
    def load_jpg_image(self, file_path):
        img = cv2.imread(file_path)  # Load JPG image
        return img

    # Convert image to grayscale
    def convert_to_grayscale(self, img):
        if len(img.shape) == 3:  # If image is in color (3 channels)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img  # Already in grayscale
        return img_gray

    # Compute SIFT descriptors for two images
    def compute_sift_descriptors(self, img1, img2):
        sift = cv2.SIFT_create()  # Create SIFT detector

        # Detect keypoints and compute descriptors for both images
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        return kp1, des1, kp2, des2

    # Match descriptors using FLANN-based matcher
    def match_descriptors(self, des1, des2):
        # FLANN parameters
        index_params = dict(algorithm=1, trees=10)
        search_params = dict(checks=50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        # Filter good matches using Lowe's ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.4 * n.distance:  # Lowe's ratio test
                good_matches.append(m)

        return good_matches

    # Draw matches between the two images
    def draw_matches(self, img1, kp1, img2, kp2, matches):
        img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        return img_matches