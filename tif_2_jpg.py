import os
from PIL import Image

def convert_tif_folder_to_jpg(input_folder, output_folder):
    """
    Converts all TIFF files in a folder to JPEG files.
    
    Args:
        input_folder (str): Path to the folder containing TIFF files.
        output_folder (str): Path to the folder to save JPEG files.
    """
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through all files in the input folder
    for file_name in os.listdir(input_folder):
        # Check if the file is a TIFF
        if file_name.lower().endswith(".tif"):
            input_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(output_folder, os.path.splitext(file_name)[0] + ".jpg")
            
            try:
                # Open and convert the TIFF file
                with Image.open(input_path) as img:
                    # Convert to RGB if necessary
                    if img.mode in ("RGBA", "P"):
                        img = img.convert("RGB")
                    # Save as JPEG
                    img.save(output_path, "JPEG")
                    print(f"Converted: {file_name} -> {output_path}")
            except Exception as e:
                print(f"Error converting {file_name}: {e}")

# Example usage

