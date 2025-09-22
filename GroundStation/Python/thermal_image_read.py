import numpy as np
import cv2
import os

def read_thermal_image(source_path, image_path):
    """
    Reads a thermal image from the specified path and returns it as a numpy array.
    
    Args:
        image_path (str): The path to the thermal image file.
        
    Returns:
        np.ndarray: The thermal image as a numpy array, or None if the image cannot be read.
    """
    if not os.path.exists(image_path):
        print(f"Image path {image_path} does not exist.")
        return None
    
    if not os.path.isfile(source_path):
        print(f"Source path {source_path} does not exist or is not a file.")
        return None
    
    source_image = cv2.imread(source_path, cv2.IMREAD_COLOR)
    print(f"Source image shape: {source_image.shape}, dtype: {source_image.dtype}")
    cv2.imshow("Source Image", cv2.resize(source_image, (source_image.shape[1]//2, source_image.shape[0]//2)))
    source_dims = source_image.shape
    
    # Read the image
    thermal_array = np.fromfile(image_path, dtype=np.float32)
    thermal_array = np.reshape(thermal_array, (source_dims[0]//2, source_dims[1]//2))

    print(f"Image read from {image_path}, shape: {thermal_array.shape}")
    print(thermal_array.max(), thermal_array.min())
    print(thermal_array.mean())
    thermal_array_bin = np.where(thermal_array > 30, 255, 0).astype(np.uint8)  # Thresholding for visualization

    contours, _ = cv2.findContours(thermal_array_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pos_list = []

    if contours is not None:
        for contour in contours:
            if cv2.contourArea(contour) > 100:
                x, y, w, h = cv2.boundingRect(contour)
                source_image = cv2.rectangle(source_image, (2*x, 2*y), (2*x + 2*w, 2*y + 2*h), (0, 255, 0), 2)
                pos={}
                pos["x"] = (x + x+w)*0.5
                pos["y"] = y
                pos_list.append(pos)
                
            else:
                continue
    else:
        print("No contours found in the thermal image.")

    cv2.imshow("Thermal Image", cv2.resize(source_image, (source_dims[1]//2, source_dims[0]//2)))
    cv2.waitKey(0)
    
    if thermal_array is None:
        print(f"Failed to read the image at {image_path}.")
        return None
    
    return thermal_array


# Example usage
if __name__ == "__main__":
    source_path = r"C:\development\WildbridgeFireVisionV2\WildBridge\ThermalImages\thermal_image_20250922_091519.jpg"
    image_path = r"C:\development\WildbridgeFireVisionV2\WildBridge\RawImages\raw_image_20250919_125715.raw"  # Replace with your image path
    thermal_image = read_thermal_image(source_path, image_path)
    
    if thermal_image is not None:
        print("Thermal image successfully read and displayed.")
    else:
        print("Failed to read the thermal image.")