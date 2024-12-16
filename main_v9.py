
import calculate_utils
import numpy as np
import cv2
import glob
import simplekml
import zipfile
import os
import pandas as pd
import exif_processing
import tif_2_jpg
import points_in_ortho
from tqdm import tqdm
import calculate_utils
import exif_processing
import user_utils
from pykml import parser
import json
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import folium



dist_coef = 0.1
start = 80
# start = 20
end = 228
# end = 40
k = 2

zero_angle = 0

cu = calculate_utils.Geodesy()
ep = exif_processing.Exif_Processing()


np.set_printoptions(suppress=True)


class TestMatcher:
    def __init__(self):
        self.keypoints_detector = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    def match(self, des1, des2):
        matches_knn = self.matcher.knnMatch(des1, des2, k=2)
        good_matches = self._ratio_test_for_knn(matches_knn)
        return good_matches

    @staticmethod
    def _ratio_test_for_knn(matches_knn):
        good_matches = []
        for m, n in matches_knn:
            if m.distance < 0.75*n.distance:
                good_matches.append(m)
        return good_matches

    def detect_and_compute(self, reference_gray, given_gray):
        self.kp1, self.des1 = self.keypoints_detector.detectAndCompute(reference_gray, None)
        self.kp2, self.des2 = self.keypoints_detector.detectAndCompute(given_gray, None)

    def run(self, reference_gray, given_gray):
        self.detect_and_compute(reference_gray, given_gray)
        return self.match(self.des1, self.des2)


def points_arrays(matches, kp1, kp2):
    
    #:param matches: List(cv2.DMatch)
    #:param kp1: List(cv2.KeyPoint)
    #:param kp2: List(cv2.KeyPoint)
    #:return: (np.ndarray, np.ndarray)
    
    points_1 = np.zeros((len(matches), 2), dtype=np.float32)
    points_2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points_1[i, :] = kp1[match.queryIdx].pt
        points_2[i, :] = kp2[match.trainIdx].pt

    return points_1, points_2

def decompose_homography(H):
    """
    Декомпозиция матрицы гомографии на поворот, смещение и масштабирование.

    Args:
        H: 3x3 numpy array, матрица гомографии.

    Returns:
        angle: угол поворота (в градусах).
        translation: (tx, ty) смещения по x и y.
        scale: общий коэффициент масштабирования.
    """
    # Извлечение смещения
    tx = H[0, 2]
    ty = H[1, 2]

    # Нормализация первого столбца для извлечения вращения
    scale = np.sqrt(H[0, 0]**2 + H[1, 0]**2)
    r11 = H[0, 0] / scale
    r21 = H[1, 0] / scale

    # Угол поворота
    angle = np.arctan2(r21, r11)
    angle_deg = np.degrees(angle)

    return angle_deg, (tx, ty), scale


class Transform:
    def __init__(self):
        self.transform_matrix = None
        self.ransac_mask = None


class AffineTransform(Transform):
    def run(self, points_1, points_2, given_img, reference_img):
        self.transform_matrix, self.ransac_mask = cv2.estimateAffinePartial2D(points_1, points_2, method=cv2.RANSAC)
        aligned_reference = cv2.warpAffine(reference_img, self.transform_matrix, (given_img.shape[1], given_img.shape[0]))
        return aligned_reference

    @property
    def transform_matrix_inv(self):
        m = np.vstack([self.transform_matrix, [0, 0, 1]])
        m_inv = np.linalg.inv(m)
        return m_inv

    def inverse_transform(self, given_image, reference_image):
        aligned_givenimg = cv2.warpAffine(given_image, self.transform_matrix_inv[:2], (reference_image.shape[1], reference_image.shape[0]))
        return aligned_givenimg


class HomographyTransform(Transform):
    def run(self, points_1, points_2, given_img, reference_img):
        self.transform_matrix, self.ransac_mask = cv2.findHomography(points_1, points_2, cv2.RANSAC)
        aligned_reference = cv2.warpPerspective(reference_img, self.transform_matrix, (given_img.shape[1], given_img.shape[0]))
        return aligned_reference

    @property
    def transform_matrix_inv(self):
        return np.linalg.inv(self.transform_matrix)

    def inverse_transform(self, given_image, reference_image):
        aligned_givenimg = cv2.warpPerspective(given_image, self.transform_matrix_inv, (reference_image.shape[1], reference_image.shape[0]))
        return aligned_givenimg

def show_big_image(img, k = 10):
    return cv2.resize(img, (img.shape[1]//k, img.shape[0]//k))

def show_aligning(given_image, aligned_reference):
    return cv2.addWeighted(given_image, 0.5, aligned_reference, 0.5, 0)

def normalize(positions):
    positions_ = positions - [np.min(positions[:,0], axis=0), np.min(positions[:,1], axis=0)]
    return positions_ / np.max(positions)

def view_positions(positions_normalized, image_size = 500, padding = [10, 10], thickness = 2, radius = 2):
    plot_coordinates = (positions_normalized * image_size).astype(int) + padding
    plot_size = np.array([np.max(plot_coordinates[:,0], axis=0), np.max(plot_coordinates[:,1], axis=0)]) + padding
    cols, rows = plot_size
    plot_img = np.ones([rows, cols, 3])

    for i in range(len(plot_coordinates)-1):
        p1 = plot_coordinates[i]
        p2 = plot_coordinates[i+1]

        cv2.circle(plot_img, p1, radius, (200, 230, 75), thickness)
        cv2.arrowedLine(plot_img, p1, p2, (200, 230, 75), thickness)
    cv2.circle(plot_img, plot_coordinates[-1], radius, (200, 230, 75), thickness)

    return plot_img

def match_points(given_matching_img, reference_image, k = 6):
    reference_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    # given_gray = cv2.cvtColor(given_image, cv2.COLOR_BGR2GRAY)

    if k > 1:
        reference_matching_img = cv2.resize(reference_gray,
                                            (reference_gray.shape[1] // k, reference_gray.shape[0] // k))
        # given_matching_img = cv2.resize(given_gray, (given_gray.shape[1] // k, given_gray.shape[0] // k))
    else:
        reference_matching_img = reference_gray

    m = TestMatcher()

    matches = m.run(reference_matching_img, given_matching_img)
    points_1, points_2 = points_arrays(matches, m.kp1, m.kp2)

    points_1, points_2 = points_1 * k, points_2 * k
    return points_1, points_2, reference_matching_img


def convert_to_wgs(vectors_set_m, lat_start, lon_start, df):
    coordinates = [{"name": str(0), "description": "__", "coords": (lat_start, lon_start)}]
    lat = lat_start
    lon = lon_start
    prev_lat = lat
    prev_lon = lon
    
    print(df.head(15))

    for i in range(len(vectors_set_m)):
        if i > 0:
            relative_distance = cu.calculate_distance_xy(x1 = vectors_set_m[i-1][0], y1 = vectors_set_m[i-1][1], x2 = vectors_set_m[i][0], y2 = vectors_set_m[i][1]) * dist_coef
            relative_angle = cu.get_2d_course_angle_image_coord_sys(x1 = vectors_set_m[i-1][0], y1 = vectors_set_m[i-1][1], x2 = vectors_set_m[i][0], y2 = vectors_set_m[i][1])
            new_lat, new_lon = cu.get_lat_lon_by_course_and_distance(lat = lat, lon = lon, l = relative_distance, azimuth = relative_angle)
            

            coordinates.append({"name": str(i), "description": "__", "coords": (new_lat, new_lon)})
            df.loc[i, 'lat'] = new_lat
            df.loc[i, 'lon'] = new_lon

            lat = new_lat
            lon = new_lon

            wgs84_dist = cu.calculate_distance_wgs84(lat1 = prev_lat, lon1 = prev_lon, lat2 = lat, lon2 = lon)
            wgs84_wear_angle = cu.calculate_wear_angle(lat1 = prev_lat, lon1 = prev_lon, lat2 = lat, lon2 = lon, zero_angle = zero_angle)
            if relative_distance > 0:
                m_in_pixel = wgs84_dist / relative_distance

            else:
                m_in_pixel = 0

            
            df.loc[i, 'wear'] = wgs84_wear_angle
            df.loc[i, 'm_in_pixel'] = m_in_pixel
            df.loc[i, 'wgs84_dist'] = wgs84_dist
            df.loc[i, 'relative_distance'] = relative_distance

            prev_lat = lat
            prev_lon = lon

    return coordinates



lon_start = 50.04541
lat_start = 30.94354


cu = calculate_utils.Geodesy()


#paths = list(glob.glob('./img15_1/1_layer/1/*.tif'))
paths = list(glob.glob('E:/farsightvision/68/1st_layer_part_B/*.tif'))


paths.sort(reverse=True)


# init
given_image = cv2.imread(paths[start])
given_gray = cv2.cvtColor(given_image, cv2.COLOR_BGR2GRAY)
given_matching_img = cv2.resize(given_gray, (given_gray.shape[1] // k, given_gray.shape[0] // k))

y, x, _ = np.array(given_image.shape)//2
points = [[[x, y]]]
points_arr = np.array(points).astype(float)
center1 = np.array((x, y))

# vector version
vector = np.array([x, y, 1])
vectors = {} # (start, end) : [vectors_set]
vectors_set = [vector[:-1]]

# matrix version
matrices_relative = []
matrices_absolute = []
matrix = np.identity(3)
vectors_set_m = [vector[:-1]]


df = pd.DataFrame({'paths': paths})

for i in range(start + 1, end):
    try:
        print("i: ", i)
        print(paths[i])
        reference_image = cv2.imread(paths[i])

        # try:
        points_1, points_2, reference_matching_img = match_points(given_matching_img, reference_image, k)
        t = AffineTransform()
        # t = HomographyTransform()
        aligned_reference = t.run(points_1, points_2, given_image, reference_image)
        aligning_image = show_aligning(given_image, aligned_reference)

        M = np.vstack([t.transform_matrix, [0,0,1]])
        # M = t.transform_matrix

        matrices_relative.append(M)
        matrix_absolute = matrix @ M
        matrices_absolute.append(matrix_absolute)
        matrix = matrix_absolute

        points_inv = cv2.perspectiveTransform(points_arr, M)
        moved_points = points_inv.astype(int)
        center2 = moved_points.reshape(-1)

        # img_res = given_image.copy()
        img_res = aligning_image
        cv2.circle(img_res, center1, 10, (200, 230, 75), 50)
        cv2.circle(img_res, center2, 10, (75, 230, 200), 50)
        cv2.line(img_res, (x, 0), (x, given_image.shape[0]-1), (0, 0, 200), 5)
        cv2.arrowedLine(img_res, center1, center2, (200, 230, 75), 20)

        # cv2_imshow()
        center_plot = show_big_image(aligning_image, k=3)

        # vector version
        vector_new = M @ vector
        vectors_set.append(vector_new[:-1].astype(int))
        vector = vector_new

        vector_m = (matrix_absolute @ np.array([x, y, 1]))[:2]
        vectors_set_m.append(vector_m)

        positions_normalized = normalize(np.array(vectors_set_m))
        positions_plot = view_positions(positions_normalized)
        y_size = center_plot.shape[0]
        positions_plot = cv2.resize(positions_plot, (y_size, y_size))
        print(center2, (M @ np.array([x, y, 1])).astype(int)[:2])
        print(vector_new.astype(int)[:2], vector_m.astype(int))

        cv2.imwrite("./result_way.JPG", np.hstack([center_plot, positions_plot]))


        given_image = reference_image
        given_matching_img = reference_matching_img

    except Exception as ex:
        print(" [error] ", ex)



points = convert_to_wgs(vectors_set_m = vectors_set_m, lat_start = lat_start, lon_start = lon_start, df = df)

print(points)

# Create a KML object
kml = simplekml.Kml()

# Add points to the KML
for point in points:
    pnt = kml.newpoint(name=point["name"], description=point["description"], coords=[point["coords"]])

# Save to a temporary KML file
kml_file = "output.kml"
kml.save(kml_file)

# Convert KML to KMZ (compress KML file)
kmz_file = "output.kmz"
with zipfile.ZipFile(kmz_file, "w", zipfile.ZIP_DEFLATED) as kmz:
    kmz.write(kml_file, arcname=os.path.basename(kml_file))

# Clean up the temporary KML file
os.remove(kml_file)

print(f"KMZ file created: {kmz_file}")


print(df.head(100))



#ep = exif_processing.Exif_Processing()
#'E:/farsightvision/68/1st_layer/*.tif'
#points_in_ortho.recover_image_centers(path = './img15_1/1_layer/1/', df = df)

#input_folder = "./img15_1/1_layer/1/"
#output_folder = "./img15_1/test_1_jpg/"

#tif_2_jpg.convert_tif_folder_to_jpg(input_folder, output_folder)

#for index, row in df.iterrows():
#    try:
#        ep.set_gps_data(image_path = row['paths'], output_path = row['paths'].replace("img15_1/1_layer/1", "test_1"), lat = row['lon'], lng = row['lat'])
#        
#    except:
#        pass






'''
# Функция для извлечения GPS-координат из EXIF
def get_gps_data(image_path):
    try:
        image = Image.open(image_path)
        exif_data = image._getexif()
        if not exif_data:
            return None

        gps_data = {}
        for tag, value in exif_data.items():
            decoded = TAGS.get(tag)
            if decoded == "GPSInfo":
                for t in value:
                    sub_decoded = GPSTAGS.get(t)
                    gps_data[sub_decoded] = value[t]

        if "GPSLatitude" in gps_data and "GPSLongitude" in gps_data:
            lat = gps_data["GPSLatitude"]
            lon = gps_data["GPSLongitude"]
            lat_ref = gps_data["GPSLatitudeRef"]
            lon_ref = gps_data["GPSLongitudeRef"]

            # Преобразование в градусы
            latitude = convert_to_degrees(lat)
            if lat_ref != "N":
                latitude = -latitude

            longitude = convert_to_degrees(lon)
            if lon_ref != "E":
                longitude = -longitude

            return latitude, longitude
    except Exception as e:
        print(f"Error processing file {image_path}: {e}")
    return None

# Преобразование координат в градусы
def convert_to_degrees(value):
    d, m, s = value
    return d + (m / 60.0) + (s / 3600.0)

# Путь к папке с фотографиями
folder_path = "E:/farsightvision/68/res/"

# Список всех фотографий в папке
photos = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg'))]

# Создаем карту
map = folium.Map(location=[0, 0], zoom_start=2)

# Добавляем метки с фотографиями
for photo in photos:
    coords = get_gps_data(photo)
    if coords:
        folium.Marker(
            location=coords,
            popup=f"<img src='{photo}' width='200'>",  # Показываем фото в всплывающем окне
        ).add_to(map)
    else:
        print(f"No GPS data found in {photo}")

# Сохраняем карту
output_map = os.path.join(folder_path, "map_with_photos.html")
map.save(output_map)
print(f"Карта сохранена: {output_map}")

'''


'''

# Определяем колонки для DataFrame
df = pd.DataFrame(columns=['name_chunk', 'lat_chunk', 'lon_chunk', 'small_chunk', 'images', 'centers'])

# Путь к папке с KML файлами
folder_path = "./kml/"
kml_file_paths = [file for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))]

print("kml_file_paths: ", kml_file_paths)

for file_path in range(len(kml_file_paths)):
    # Парсинг KML файла
    with open(folder_path + kml_file_paths[file_path], 'r') as file:
        root = parser.parse(file).getroot()

    # Итерация по Placemark элементам
    namespace = {"kml": "http://www.opengis.net/kml/2.2"}
    placemarks = root.findall(".//kml:Placemark", namespaces=namespace)

    for placemark in placemarks:
        # Извлечение имени
        name = placemark.find(".//kml:name", namespaces=namespace)
        name_text = name.text if name is not None else "Unnamed"

        # Извлечение координат
        coordinates = placemark.find(".//kml:coordinates", namespaces=namespace)
        if coordinates is not None:
            # Преобразование координат в список
            coords_list = [float(coord) for coord in coordinates.text.strip().split(",")]
            lon_chunk, lat_chunk, *_ = coords_list  # Извлекаем долготу, широту (игнорируем высоту, если есть)
        else:
            lon_chunk, lat_chunk = None, None

        print(f"Name: {kml_file_paths[file_path]} ({name_text}), coordinates: {coordinates.text if coordinates is not None else 'No geometry'}")

        # Добавление данных в DataFrame
        df.loc[len(df)] = [kml_file_paths[file_path], lat_chunk, lon_chunk, None, None, None]

# Проверяем итоговый DataFrame
#print(df.head(10))

#df.to_csv("result.csv", index=False)


folder_path = "E:/farsightvision/68/1st_layer_part_B/"  
json_files = [f for f in os.listdir(folder_path) if f.endswith(".json")]

print(len(json_files))

#json_files_group_A = json_files[0:58]
#json_files_group_B = json_files[60:105]
#json_files_group_C = json_files[114:175]
#json_files_group_D = json_files[189:229]

#print("json_files_group_A: ", len(json_files_group_A))
#print("json_files_group_B: ", len(json_files_group_B))
#print("json_files_group_C: ", len(json_files_group_C))
#print("json_files_group_D: ", len(json_files_group_D))

#json_files_final = json_files_group_A + json_files_group_B + json_files_group_C + json_files_group_D
json_files_final = json_files

print("json_files_final: ", len(json_files_final))

print("=========================================")
print(f" [info] Creation images with coordinates")

for i in tqdm(range(len(json_files))):
    try:
        with open(folder_path + json_files[i], 'r') as file:
            images = str(json.load(file)['images'])
            df.loc[i, 'images'] = images

            current_chunk_lat = df.loc[i, "lat_chunk"]
            current_chunk_lon = df.loc[i, "lon_chunk"]
            
            #previous_chunk_lat = df.loc[i-1, "lat_chunk"]
            #previous_chunk_lon = df.loc[i-1, "lon_chunk"]

            #dist_between_prev_and_cur_chunks_centers = cu.calculate_distance_wgs84(lat1 = previous_chunk_lat, lon1 = previous_chunk_lon, lat2 = current_chunk_lat, lon2 = current_chunk_lon) 
            #wear_angle_between_prev_and_cur_chunks_centers = cu.calculate_wear_angle(lat1 = previous_chunk_lat, lon1 = previous_chunk_lon, lat2 = current_chunk_lat, lon2 = current_chunk_lon, zero_angle = 0)   
            #l1_lat, l1_lon = cu.get_lat_lon_by_course_and_distance(lat = current_chunk_lat, lon = current_chunk_lon, l = 0.835*dist_between_prev_and_cur_chunks_centers, azimuth = wear_angle_between_prev_and_cur_chunks_centers)
            #l2_lat, l2_lon = cu.get_lat_lon_by_course_and_distance(lat = current_chunk_lat, lon = current_chunk_lon, l = 0.501*dist_between_prev_and_cur_chunks_centers, azimuth = wear_angle_between_prev_and_cur_chunks_centers)
            #l3_lat, l3_lon = cu.get_lat_lon_by_course_and_distance(lat = current_chunk_lat, lon = current_chunk_lon, l = 0.167*dist_between_prev_and_cur_chunks_centers, azimuth = wear_angle_between_prev_and_cur_chunks_centers)
            #print(l1_lat, l1_lon)
            #print(l2_lat, l2_lon)
            #print(l3_lat, l3_lon)


            if i != (len(json_files)-1):

                next_chunk_lat = df.loc[i+1, "lat_chunk"]
                next_chunk_lon = df.loc[i+1, "lon_chunk"]

                next_chunk_lat = df.loc[i, "lat_chunk"]
                curent_chunk_lon = df.loc[i, "lon_chunk"]
                
                dist_between_cur_and_next_chunks_centers = cu.calculate_distance_wgs84(lat1 = current_chunk_lat, lon1 = current_chunk_lon, lat2 = next_chunk_lat, lon2 = next_chunk_lon) 
                wear_angle_between_cur_and_next_chunks_centers = cu.calculate_wear_angle(lat1 = current_chunk_lat, lon1 = current_chunk_lon, lat2 = next_chunk_lat, lon2 = next_chunk_lon, zero_angle = 0)

                r1_lat, r1_lon = cu.get_lat_lon_by_course_and_distance(lat = current_chunk_lat, lon = current_chunk_lon, l = 0.167*dist_between_cur_and_next_chunks_centers, azimuth = wear_angle_between_cur_and_next_chunks_centers)
                r2_lat, r2_lon = cu.get_lat_lon_by_course_and_distance(lat = current_chunk_lat, lon = current_chunk_lon, l = 0.501*dist_between_cur_and_next_chunks_centers, azimuth = wear_angle_between_cur_and_next_chunks_centers)
                r3_lat, r3_lon = cu.get_lat_lon_by_course_and_distance(lat = current_chunk_lat, lon = current_chunk_lon, l = 0.835*dist_between_cur_and_next_chunks_centers, azimuth = wear_angle_between_cur_and_next_chunks_centers)
                
                centers = str([r1_lat, r1_lon, r2_lat, r2_lon, r3_lat, r3_lon])
                df.loc[i, 'centers'] = centers

                #print(df.iloc[i]['images'])

                #print(r1_lat, r1_lon)
                #print(r2_lat, r2_lon)
                #print(r3_lat, r3_lon)
                
                #print("=========================================")

                
                ep.set_gps_data(image_path = "E:/farsightvision/68/ADTi_F016/" + df.iloc[i]['images'].split(" ")[3], output_path = "E:/farsightvision/68/res/" + df.iloc[i]['images'].split(" ")[3], lat = r1_lat, lng = r1_lon)
                ep.set_gps_data(image_path = "E:/farsightvision/68/ADTi_F016/" + df.iloc[i]['images'].split(" ")[4], output_path = "E:/farsightvision/68/res/" + df.iloc[i]['images'].split(" ")[4], lat = r2_lat, lng = r2_lon)
                ep.set_gps_data(image_path = "E:/farsightvision/68/ADTi_F016/" + df.iloc[i]['images'].split(" ")[5], output_path = "E:/farsightvision/68/res/" + df.iloc[i]['images'].split(" ")[5], lat = r3_lat, lng = r3_lon)
                

        df.loc[i, 'small_chunk'] = json_files_final[i]

    except Exception as ex:
        print(ex)

print(df.head(15))

'''