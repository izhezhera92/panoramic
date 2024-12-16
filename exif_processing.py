import piexif
from PIL import Image

class Exif_Processing(object):
    def __init__(self):
        pass

    def get_coordinates_list(self, path = None):
        if path is None:
            path = self.image_path

        files_list = self.__get_files_list(path = path)
        coordinates = []
        print("[INFO] Get all gps coordinates from exif ..")

        for i in tqdm(range(len(files_list))):
            image_path = self.image_path + files_list[i]
            latitude, longitude = self.get_gps_data(image_path)

            coordinates.append((longitude, latitude))
            #if latitude and longitude:
            #    print(f"Lat = {latitude}, Lon = {longitude}")
            #else:
            #    print("No GPS data found.")
        return coordinates

    def __set_all_centers_coordinates(self, path, wgs_points):
        image_list = self.__get_files_list(path)
        print("[INFO] Setting coordinates ..")
        for i in tqdm(range(len(image_list))):
            image_path = self.image_path + image_list[i]
            lat, lon = wgs_points[i]
            self.set_gps_data(image_path, image_path, lat, lon)
        print("[INFO] Coordinates set!")
        return 0


    def to_exif_gps_format(self, coordinate):
        """
        Convert a GPS coordinate (e.g., latitude or longitude) to EXIF format.
        """
        degrees = int(coordinate)
        minutes = int((coordinate - degrees) * 60)
        seconds = int(((coordinate - degrees) * 60 - minutes) * 60 * 10000)  # multiplied by 10000 to keep precision
        
        return (degrees, 1), (minutes, 1), (seconds, 10000)

    def convert_to_degrees(self, value, ref):
        # Convert GPS coordinates from (degrees, minutes, seconds) to decimal
        d = value[0][0] / value[0][1]
        m = value[1][0] / value[1][1]
        s = value[2][0] / value[2][1]
        
        decimal = d + (m / 60.0) + (s / 3600.0)
        
        if ref in [b'S', b'W']:
            decimal = -decimal  # South and West should be negative values
        
        return decimal

    def set_gps_data(self, image_path, output_path, lat, lng):
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
        exif_lat = self.to_exif_gps_format(abs(lat))
        exif_lng = self.to_exif_gps_format(abs(lng))
        
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
        return 0

    def get_gps_data(self, path, round_val = 6):
        # Load the image and extract EXIF data
        img = Image.open(path)
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
                latitude = self.convert_to_degrees(gps_latitude, lat_ref)
                longitude = self.convert_to_degrees(gps_longitude, lon_ref)
            else:
                latitude, longitude = None, None
        else:
            latitude, longitude = None, None


        return round(latitude, round_val), round(longitude, round_val)

