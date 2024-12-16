'''
# READ EXIF DATA GPS AND FOCAL LENGHT FROM IMAGE 

import piexif
from PIL import Image

def get_gps_and_focal_length(image_path):
    # Load the image and extract EXIF data
    img = Image.open(image_path)
    exif_data = piexif.load(img.info['exif'])
    
    # Initialize GPS and focal length variables
    gps_info = None
    focal_length = None
    
    # Extract GPS data if available
    if "GPS" in exif_data:
        gps_info = exif_data['GPS']
        
        # GPSLatitude and GPSLongitude
        gps_latitude = gps_info.get(piexif.GPSIFD.GPSLatitude)
        gps_longitude = gps_info.get(piexif.GPSIFD.GPSLongitude)
        
        # GPSLatitudeRef and GPSLongitudeRef (N/S, E/W)
        lat_ref = gps_info.get(piexif.GPSIFD.GPSLatitudeRef)
        lon_ref = gps_info.get(piexif.GPSIFD.GPSLongitudeRef)

        if gps_latitude and gps_longitude:
            # Convert to degrees
            latitude = convert_to_degrees(gps_latitude, lat_ref)
            longitude = convert_to_degrees(gps_longitude, lon_ref)
        else:
            latitude, longitude = None, None
    else:
        latitude, longitude = None, None

    # Extract focal length if available
    if "Exif" in exif_data:
        exif_info = exif_data["Exif"]
        focal_length_data = exif_info.get(piexif.ExifIFD.FocalLength)
        
        if focal_length_data:
            focal_length = focal_length_data[0] / focal_length_data[1]  # Focal length is a rational number
    
    return latitude, longitude, focal_length

def convert_to_degrees(value, ref):
    # Convert GPS coordinates from (degrees, minutes, seconds) to decimal
    d = value[0][0] / value[0][1]
    m = value[1][0] / value[1][1]
    s = value[2][0] / value[2][1]
    
    decimal = d + (m / 60.0) + (s / 3600.0)
    
    if ref in [b'S', b'W']:
        decimal = -decimal  # South and West should be negative values
    
    return decimal

# Example usage:
image_path = './DJI_0490.JPG'
latitude, longitude, focal_length = get_gps_and_focal_length(image_path)

if latitude and longitude:
    print(f"GPS Coordinates: Latitude = {latitude}, Longitude = {longitude}")
else:
    print("No GPS data found.")

if focal_length:
    print(f"Focal Length: {focal_length} mm")
else:
    print("No Focal Length data found.")
'''


'''
import math

def calculate_angle_of_view(sensor_size_mm, focal_length_mm):
    """
    Calculate the angle of view (FoV) given the sensor size and focal length.
    
    :param sensor_size_mm: Size of the camera sensor in mm (can be width, height, or diagonal).
    :param focal_length_mm: Focal length of the lens in mm.
    :return: Angle of view in degrees.
    """
    # Calculate the angle of view using the formula
    angle_of_view_radians = 2 * math.atan(sensor_size_mm / (2 * focal_length_mm))
    
    # Convert radians to degrees
    angle_of_view_degrees = math.degrees(angle_of_view_radians)
    
    return angle_of_view_degrees

# Example usage

# Sensor dimensions (in mm) for full-frame (36mm x 24mm)
sensor_width_mm = 36.0
sensor_height_mm = 24.0

# Focal length of the lens (in mm)
focal_length_mm = 50  # Example: 50mm lens

# Calculate the horizontal and vertical angle of view
horizontal_angle = calculate_angle_of_view(sensor_width_mm, focal_length_mm)
vertical_angle = calculate_angle_of_view(sensor_height_mm, focal_length_mm)

print(f"Horizontal Angle of View: {horizontal_angle:.2f} degrees")
print(f"Vertical Angle of View: {vertical_angle:.2f} degrees")

'''




'''
import piexif
from PIL import Image

def convert_to_rational(number):
    """Converts a float to a rational number (tuple of numerator and denominator)."""
    frac = number.as_integer_ratio()
    return frac[0], frac[1]

def deg_to_dms_rational(deg_float):
    """Convert decimal degrees to degrees, minutes, seconds in rational format."""
    degrees = int(deg_float)
    minutes = int((deg_float - degrees) * 60)
    seconds = (deg_float - degrees - minutes/60) * 3600
    return [(degrees, 1), (minutes, 1), convert_to_rational(seconds)]

def insert_gps_data(image_path, lat, lon, output_image_path):
    # Convert latitude and longitude to degrees, minutes, and seconds
    lat_dms = deg_to_dms_rational(abs(lat))
    lon_dms = deg_to_dms_rational(abs(lon))

    # Define the GPS EXIF tags
    gps_ifd = {
        piexif.GPSIFD.GPSLatitudeRef: 'N' if lat >= 0 else 'S',
        piexif.GPSIFD.GPSLatitude: lat_dms,
        piexif.GPSIFD.GPSLongitudeRef: 'E' if lon >= 0 else 'W',
        piexif.GPSIFD.GPSLongitude: lon_dms,
        piexif.GPSIFD.GPSAltitude: (0, 1),  # Altitude (0 for sea level)
        piexif.GPSIFD.GPSAltitudeRef: 0  # 0 = above sea level, 1 = below sea level
    }

    # Load image and extract existing EXIF data
    img = Image.open(image_path)
    exif_dict = piexif.load(img.info['exif']) if 'exif' in img.info else {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}}

    # Update the GPS IFD in the EXIF data
    exif_dict['GPS'] = gps_ifd

    # Convert back the EXIF dictionary to binary format
    exif_bytes = piexif.dump(exif_dict)

    # Save the image with the new EXIF data
    img.save(output_image_path, exif=exif_bytes)

    print(f"GPS data written to {output_image_path}")

# Example usage
image_path = './DJI_0490_copy.JPG'
output_image_path = 'output_image_with_gps.jpg'
latitude = 37.7749  # Positive for North, negative for South
longitude = -122.4194  # Positive for East, negative for West

insert_gps_data(image_path, latitude, longitude, output_image_path)

'''

'''

import piexif
from PIL import Image

def to_exif_gps_format(coordinate):
    """
    Convert a GPS coordinate (e.g., latitude or longitude) to EXIF format.
    """
    degrees = int(coordinate)
    minutes = int((coordinate - degrees) * 60)
    seconds = int(((coordinate - degrees) * 60 - minutes) * 60 * 10000)  # multiplied by 10000 to keep precision
    
    return (degrees, 1), (minutes, 1), (seconds, 10000)

def add_gps_data(image_path, output_path, lat, lng):
    """
    Add GPS data to an image's EXIF metadata.
    
    Parameters:
    - image_path: Path to the input image.
    - output_path: Path to save the image with added GPS EXIF data.
    - lat: Latitude in decimal format (e.g., 37.7749).
    - lng: Longitude in decimal format (e.g., -122.4194).
    """
    # Open the image
    image = Image.open(image_path)
    
    # Load existing EXIF data (if any) or initialize a new dictionary
    exif_data = piexif.load(image.info.get('exif', piexif.dump({})))
    
    # Convert latitude and longitude to EXIF format
    exif_lat = to_exif_gps_format(abs(lat))
    exif_lng = to_exif_gps_format(abs(lng))
    
    # Set GPS data in EXIF
    gps_ifd = {
        piexif.GPSIFD.GPSLatitudeRef: b'N' if lat >= 0 else b'S',
        piexif.GPSIFD.GPSLatitude: exif_lat,
        piexif.GPSIFD.GPSLongitudeRef: b'E' if lng >= 0 else b'W',
        piexif.GPSIFD.GPSLongitude: exif_lng,
    }
    
    # Add GPS data to the EXIF dictionary
    exif_data['GPS'] = gps_ifd
    
    # Insert EXIF data into the image and save it
    exif_bytes = piexif.dump(exif_data)
    image.save(output_path, exif=exif_bytes)

# Example usage
image_path = 'DJI_0490_copy.JPG'        # Input image path
output_path = 'DJI_0490_copy_new.JPG'  # Output image path
latitude = 37.7749              # Latitude in decimal degrees
longitude = -122.4194           # Longitude in decimal degrees

add_gps_data(image_path, output_path, latitude, longitude)

'''

'''

import rasterio
from rasterio.warp import transform

def pixel_to_wgs84(geotiff_path, pixel_x, pixel_y):
    """
    Get WGS84 coordinates (latitude, longitude) for a specific pixel in a GeoTIFF image.
    
    Parameters:
    - geotiff_path: Path to the GeoTIFF file.
    - pixel_x: X coordinate (column) of the pixel.
    - pixel_y: Y coordinate (row) of the pixel.
    
    Returns:
    - (latitude, longitude): The geographic coordinates in WGS84 format.
    """
    # Open the GeoTIFF file
    with rasterio.open(geotiff_path) as dataset:
        # Get the pixel's geographic coordinates (in the dataset's CRS)
        point = dataset.transform * (pixel_x, pixel_y)
        
        # Extract the source CRS (from the GeoTIFF metadata)
        src_crs = dataset.crs
        
        # Transform the coordinates to WGS84 (EPSG:4326)
        lon, lat = transform(src_crs, 'EPSG:4326', [point[0]], [point[1]])
        
        # Return latitude and longitude
        return lat[0], lon[0]

# Example usage
geotiff_path = './odm_orthophoto.tif'  # Path to your GeoTIFF file
pixel_x = 100                    # X coordinate (column) of the pixel
pixel_y = 200                    # Y coordinate (row) of the pixel

latitude, longitude = pixel_to_wgs84(geotiff_path, pixel_x, pixel_y)
print(f"The WGS84 coordinates of the pixel are: Latitude = {latitude}, Longitude = {longitude}")
'''

'''
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Загрузим два изображения (предыдущее и текущее)
img1 = cv2.imread('2.jpg')  # Предыдущее изображение
img2 = cv2.imread('1.jpg')   # Текущее изображение


# Найдем ключевые точки и дескрипторы с помощью SIFT или ORB (например, ORB)
orb = cv2.ORB_create()
keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

# Соответствие ключевых точек с использованием BFMatcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)
matches = sorted(matches, key=lambda x: x.distance)

# Возьмем только лучшие соответствия для нахождения гомографии
good_matches = matches[:50]
src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# Вычисляем матрицу гомографии
H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

# Найдем центр текущего изображения (img2)
current_img_height, current_img_width = img2.shape[:2]
current_center = np.array([[current_img_width / 2, current_img_height / 2]], dtype="float32")
current_center = np.array([current_center])  # Преобразуем в форму (1, 1, 2) для cv2.perspectiveTransform

# Применяем гомографию для нахождения положения центра текущего изображения на изображении img1
transformed_center = cv2.perspectiveTransform(current_center, H)
x_transformed_center, y_transformed_center = transformed_center[0][0]

# Найдем центр предыдущего изображения (img1)
prev_img_height, prev_img_width = img1.shape[:2]
prev_center_x, prev_center_y = prev_img_width / 2, prev_img_height / 2

# Выводим координаты центров
print(f"Coordinates of the previous image's center: ({prev_center_x}, {prev_center_y})")
print(f"Coordinates of the current image's transformed center: ({x_transformed_center}, {y_transformed_center})")

# Отобразим оба изображения и отметим оба центра на итоговом изображении
plt.figure(figsize=(12, 6))

# Создаем наложенное изображение
img1_warped = cv2.warpPerspective(img2, H, (img1.shape[1], img1.shape[0]))
overlay_img = cv2.addWeighted(img1, 0.5, img1_warped, 0.5, 0)

plt.imshow(cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB))

# Отметим центры
# Центр предыдущего изображения (img1)
plt.scatter([prev_center_x], [prev_center_y], color='blue', s=100, marker='o', label='Center of Previous Image')
plt.text(prev_center_x, prev_center_y, 'Prev Center', color='blue', fontsize=12)

# Центр текущего изображения (img2), после трансформации
plt.scatter([x_transformed_center], [y_transformed_center], color='red', s=100, marker='x', label='Transformed Center of Current Image')
plt.text(x_transformed_center, y_transformed_center, 'Transformed Center', color='red', fontsize=12)

# Настройки графика
plt.legend()
plt.title("Centers of Previous and Transformed Current Images")
plt.show()

'''

'''

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Загрузим изображения (по порядку)

def __get_files_list(path):
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    return [f for f in os.listdir(path) if f.lower().endswith(supported_formats)]


def __create_image_list(lst, path):
    img_list = []
    for i in range(len(lst)):
        new_image = cv2.imread(path + lst[i])
        #new_image = cv2.resize(new_image, (0, 0), fx = 1.0, fy = 1.0)
        img_list.append(new_image)
    return img_list



file_list = __get_files_list(path = "./img16/")


images = __create_image_list(lst = file_list, path = "./img16/")

# Check if all images are loaded
for i, img in enumerate(images):
    if img is None:
        raise ValueError(f"Image {i + 1} not found or could not be loaded.")

# Parameters for storing centers and homographies
centers = []
homographies = []

# Store the center of the first image (base)
base_img_height, base_img_width = images[0].shape[:2]
base_center = (base_img_width / 2, base_img_height / 2)
centers.append(base_center)

# Initialize ORB and BFMatcher
orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Create a blank image for the accumulated result with extra space
#accumulated_image = np.zeros((base_img_height * 3, base_img_width * 3, 3), dtype=np.uint8)
accumulated_image = np.zeros((20000, 20000, 3), dtype=np.uint8)


# Position the first image in the center of the accumulated image
center_offset = (base_img_width, base_img_height)
accumulated_image[
    center_offset[1]:center_offset[1] + base_img_height,
    center_offset[0]:center_offset[0] + base_img_width
] = images[0]

# Iterate through pairs of images to compute homographies and centers
for i in range(1, len(images)):
    # Compute keypoints and descriptors for the current and previous image pair
    keypoints_prev, descriptors_prev = orb.detectAndCompute(images[i-1], None)
    keypoints_curr, descriptors_curr = orb.detectAndCompute(images[i], None)
    
    # Match the descriptors
    matches = bf.match(descriptors_prev, descriptors_curr)
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Filter matches
    if len(matches) < 10:
        raise ValueError(f"Not enough matches found between image {i} and image {i-1}.")
    
    good_matches = matches[:100]  # Use the best matches
    
    # Extract points for homography
    src_pts = np.float32([keypoints_prev[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints_curr[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # Compute the homography between the current and previous image
    H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    homographies.append(H)
    
    # Find the center of the current image
    curr_img_height, curr_img_width = images[i].shape[:2]
    curr_center = (curr_img_width / 2, curr_img_height / 2)  # Store as tuple
    
    # Transform the center of the current image into the coordinate system of the base image
    transformed_center = np.array([[curr_center[0], curr_center[1]]], dtype="float32").reshape(-1, 1, 2)
    
    for j in range(i):  # Apply all previous homographies to get the final position
        transformed_center = cv2.perspectiveTransform(transformed_center, homographies[j])
    
    # Save the transformed center as a tuple
    centers.append((transformed_center[0][0][0], transformed_center[0][0][1]))

    # Calculate the homography relative to the center position of the accumulated image
    offset_H = np.eye(3)
    offset_H[0, 2] = center_offset[0]
    offset_H[1, 2] = center_offset[1]
    
    # Full homography to apply
    if homographies:  # Check if there are any homographies to accumulate
        print(f"Accumulating homographies up to image {i}: {homographies[:i]}")
        
        if len(homographies[:i]) > 1:  # We need at least two homographies to accumulate
            full_H = offset_H @ np.linalg.multi_dot(homographies[:i][::-1])  # Apply accumulated homographies
        else:
            full_H = offset_H @ homographies[0]  # If only one homography, just use that
    else:
        full_H = offset_H  # If no previous homographies, just use the offset

    warped_image = cv2.warpPerspective(images[i], full_H, (accumulated_image.shape[1], accumulated_image.shape[0]))

    # Add the current image to the accumulated image
    mask = (warped_image > 0)  # Ignore black (zero) pixels
    accumulated_image[mask] = warped_image[mask]

# Display the final accumulated image with marked centers of all images
plt.figure(figsize=(12, 6))
plt.imshow(cv2.cvtColor(accumulated_image, cv2.COLOR_BGR2RGB))

# Mark the centers of the images
for idx, center in enumerate(centers):
    # Print center for debugging
    print(f"Center of Image {idx + 1}: {center}")
    
    x, y = center  # Now we can safely unpack
    x += center_offset[0]  # Adjust coordinates for the accumulated image
    y += center_offset[1]  # Adjust coordinates for the accumulated image
    
    plt.scatter([x], [y], s=100, label=f'Center of Image {idx + 1}')
    plt.text(x, y, f'{idx + 1}', color='white', fontsize=12)

# Graph settings
#plt.legend()
plt.title("Centers of All Images on Accumulated Image")
plt.axis('off')  # Turn off the axis
plt.show()

'''


import os
import cv2
import numpy as np

# Загрузим изображения (по порядку)
def __get_files_list(path):
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    return [f for f in os.listdir(path) if f.lower().endswith(supported_formats)]

def __create_image_list(lst, path):
    img_list = []
    for i in range(len(lst)):
        new_image = cv2.imread(os.path.join(path, lst[i]))
        img_list.append(new_image)
    return img_list

file_list = __get_files_list(path="./img16/")
images = __create_image_list(lst=file_list, path="./img16/")

# Проверка загрузки всех изображений
for i, img in enumerate(images):
    if img is None:
        raise ValueError(f"Image {i + 1} not found or could not be loaded.")

# Параметры для хранения центров и гомографий
centers = []
homographies = []

# Центрируем первое изображение
base_img_height, base_img_width = images[0].shape[:2]
base_center = (base_img_width / 2, base_img_height / 2)
centers.append(base_center)

# Инициализация ORB и BFMatcher
orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Создаём пустое изображение для накопленного результата
accumulated_image = np.zeros((20000, 20000, 3), dtype=np.uint8)
center_offset = (10000 - base_img_width // 2, 10000 - base_img_height // 2)
accumulated_image[
    center_offset[1]:center_offset[1] + base_img_height,
    center_offset[0]:center_offset[0] + base_img_width
] = images[0]

# Итерация по парам изображений для вычисления гомографий и центров
for i in range(1, len(images)):
    keypoints_prev, descriptors_prev = orb.detectAndCompute(images[i - 1], None)
    keypoints_curr, descriptors_curr = orb.detectAndCompute(images[i], None)
    
    matches = bf.match(descriptors_prev, descriptors_curr)
    matches = sorted(matches, key=lambda x: x.distance)

    if len(matches) < 10:
        raise ValueError(f"Not enough matches found between image {i} and image {i-1}.")

    good_matches = matches[:100]

    src_pts = np.float32([keypoints_prev[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints_curr[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

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

    if len(homographies[:i]) > 1:
        full_H = offset_H @ np.linalg.multi_dot(homographies[:i][::-1])
    else:
        full_H = offset_H @ homographies[0]

    warped_image = cv2.warpPerspective(images[i], full_H, (accumulated_image.shape[1], accumulated_image.shape[0]))
    mask = (warped_image > 0)
    accumulated_image[mask] = warped_image[mask]
    
# Отметка центров на накопленном изображении
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
output_filename = "accumulated_image_with_centers.jpg"
output_path = os.path.join(output_dir, output_filename)
os.makedirs(output_dir, exist_ok=True)
cv2.imwrite(output_path, accumulated_image)
print(f"Accumulated image with centers saved as '{output_path}'")
