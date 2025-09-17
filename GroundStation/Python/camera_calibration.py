import cv2
import numpy as np
import os
import glob
import json
from datetime import datetime

class CameraCalibrator:
    def __init__(self, chessboard_size=(8, 6)):
        self.chessboard_size = chessboard_size
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # Prepare object points
        self.objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        
        # Arrays to store object points and image points
        self.objpoints = []  # 3d point in real world space
        self.imgpoints = []  # 2d points in image plane
        
    def extract_frames_from_video(self, video_path, frame_interval=30):
        """Extract frames from video for calibration"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        
        print(f"Extracting frames from video: {video_path}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                frames.append(frame)
                
            frame_count += 1
            
        cap.release()
        print(f"Extracted {len(frames)} frames from video")
        return frames
    
    def find_chessboard_corners(self, images):
        """Find chessboard corners in images"""
        valid_images = []
        
        for i, img in enumerate(images):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)
            
            if ret:
                self.objpoints.append(self.objp)
                
                # Refine corner positions
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
                self.imgpoints.append(corners2)
                
                # Draw and display the corners
                img_with_corners = img.copy()
                cv2.drawChessboardCorners(img_with_corners, self.chessboard_size, corners2, ret)
                
                # Resize for display if image is too large
                height, width = img_with_corners.shape[:2]
                if width > 800:
                    scale = 800 / width
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    img_with_corners = cv2.resize(img_with_corners, (new_width, new_height))
                
                cv2.imshow(f'Chessboard Detection - Image {i+1}', img_with_corners)
                key = cv2.waitKey(500)
                
                if key == ord('q'):
                    break
                    
                valid_images.append(img)
                print(f"Found chessboard in image {i+1}")
            else:
                print(f"No chessboard found in image {i+1}")
        
        cv2.destroyAllWindows()
        return valid_images
    
    def calibrate_camera(self, images):
        """Perform camera calibration"""
        if len(self.objpoints) == 0:
            print("No valid chessboard patterns found!")
            return None, None
        
        print(f"Calibrating camera with {len(self.objpoints)} images...")
        
        gray = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)
        img_size = gray.shape[::-1]
        
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, img_size, None, None)
        
        if ret:
            print("Camera calibration successful!")
            print(f"Reprojection error: {ret}")
            return camera_matrix, dist_coeffs
        else:
            print("Camera calibration failed!")
            return None, None
    
    def save_calibration(self, camera_matrix, dist_coeffs, filename="camera_calibration.json"):
        """Save calibration results as JSON"""
        calibration_data = {
            "calibration_info": {
                "timestamp": datetime.now().isoformat(),
                "chessboard_size": list(self.chessboard_size),
                "num_images_used": len(self.objpoints)
            },
            "camera_matrix": {
                "fx": float(camera_matrix[0, 0]),
                "fy": float(camera_matrix[1, 1]),
                "cx": float(camera_matrix[0, 2]),
                "cy": float(camera_matrix[1, 2]),
                "matrix": camera_matrix.tolist()
            },
            "distortion_coefficients": {
                "k1": float(dist_coeffs[0, 0]),
                "k2": float(dist_coeffs[0, 1]),
                "p1": float(dist_coeffs[0, 2]),
                "p2": float(dist_coeffs[0, 3]),
                "k3": float(dist_coeffs[0, 4]),
                "coefficients": dist_coeffs.tolist()
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(calibration_data, f, indent=2)
        
        print(f"Calibration saved to {filename}")
        
        # Also save as NPZ for backward compatibility
        npz_filename = filename.replace('.json', '.npz')
        np.savez(npz_filename, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
        print(f"Calibration also saved to {npz_filename} for backward compatibility")

def main():
    """Interactive camera calibration tool"""
    print("="*50)
    print("        CAMERA CALIBRATION TOOL")
    print("="*50)
    print()
    
    # Get input type
    print("Select input type:")
    print("1. Video file")
    print("2. Image directory")
    input_choice = input("Enter choice (1 or 2): ").strip()
    
    if input_choice == '1':
        input_type = 'video'
        input_path = input("Enter path to video file: ").strip()
        
        if not os.path.exists(input_path):
            print(f"Error: Video file not found: {input_path}")
            return
            
        frame_interval = input("Frame interval for extraction (default 30): ").strip()
        frame_interval = int(frame_interval) if frame_interval else 30
        
    elif input_choice == '2':
        input_type = 'images'
        input_path = input("Enter path to image directory: ").strip()
        input_path = input_path.strip("\"")
        
        if not os.path.exists(input_path):
            print(f"Error: Directory not found: {input_path}")
            return
            
        frame_interval = None
        
    else:
        print("Invalid choice!")
        return
    
    # Get chessboard configuration
    print("\nChessboard Configuration:")
    print("Count the INTERNAL corners (intersection points), not squares!")
    print("Example: 8x6 grid of squares = 7x5 internal corners")
    
    chessboard_width = input("Chessboard width (internal corners, default 9): ").strip()
    chessboard_width = int(chessboard_width) if chessboard_width else 9
    
    chessboard_height = input("Chessboard height (internal corners, default 6): ").strip()
    chessboard_height = int(chessboard_height) if chessboard_height else 6
    
    chessboard_size = (chessboard_width, chessboard_height)
    
    # Save drone specific camera calibration files
    print("\n Specify drone name for which calibration is being performed (for file naming):")
    drone_name = input("Drone name: ").strip()

    if drone_name == "M300RTK_A":
        output_file = f'camera_calibration_{drone_name}.json'
    elif drone_name == "M300RTK_B":
        output_file = f'camera_calibration_{drone_name}.json'
    elif drone_name == "M4T":
        output_file = f'camera_calibration_{drone_name}.json'
    elif drone_name == "M3E":
        output_file = f'camera_calibration_{drone_name}.json'
    else:
        print("Drone name not recognized. Enter a valid name (M300RTK_A, M300RTK_B, M4T, M3E).")
        return
    
    print("\n" + "="*50)
    print("CONFIGURATION SUMMARY:")
    print("="*50)
    print(f"Input type: {input_type}")
    print(f"Input path: {input_path}")
    if input_type == 'video':
        print(f"Frame interval: {frame_interval}")
    print(f"Chessboard size: {chessboard_width}x{chessboard_height} corners")
    print(f"Output file: {output_file}")
    print("="*50)
    
    confirm = input("\nProceed with calibration? (y/n): ").strip().lower()
    if confirm != 'y' and confirm != 'yes':
        print("Calibration cancelled.")
        return
    
    # Initialize calibrator
    calibrator = CameraCalibrator(chessboard_size)
    
    # Load images
    if input_type == 'video':
        images = calibrator.extract_frames_from_video(input_path, frame_interval)
    else:
        # Load images from directory
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(input_path, ext)))
        
        if not image_files:
            print(f"No images found in directory: {input_path}")
            return
        
        # Load images
        images = []
        for img_path in image_files:
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
    
    if not images:
        print("No valid images found!")
        return
    
    print(f"\nProcessing {len(images)} images...")
    print("Press 'q' to quit early during chessboard detection")
    
    # Find chessboard corners
    valid_images = calibrator.find_chessboard_corners(images)
    
    if len(valid_images) < 3:
        print("Need at least 3 valid chessboard images for calibration!")
        return
    
    # Calibrate camera
    camera_matrix, dist_coeffs = calibrator.calibrate_camera(valid_images)
    
    if camera_matrix is not None:
        print("\nCalibration Results:")
        print("Camera Matrix:")
        print(camera_matrix)
        print("\nDistortion Coefficients:")
        print(dist_coeffs)
        
        # Save results
        calibrator.save_calibration(camera_matrix, dist_coeffs, output_file)
        
        # Display results in a readable format
        print(f"\nCalibration Summary:")
        print(f"Focal length (fx, fy): ({camera_matrix[0,0]:.2f}, {camera_matrix[1,1]:.2f})")
        print(f"Principal point (cx, cy): ({camera_matrix[0,2]:.2f}, {camera_matrix[1,2]:.2f})")
        print(f"Distortion coefficients (k1, k2, p1, p2, k3): {dist_coeffs.flatten()}")
        print(f"Number of images used: {len(calibrator.objpoints)}")
        print(f"Chessboard size: {calibrator.chessboard_size}")
        
        # Load and display the JSON for verification
        print(f"\nJSON output preview:")
        try:
            with open(output_file, 'r') as f:
                json_data = json.load(f)
            print(json.dumps(json_data, indent=2)[:500] + "..." if len(json.dumps(json_data, indent=2)) > 500 else json.dumps(json_data, indent=2))
        except Exception as e:
            print(f"Error reading JSON file: {e}")

if __name__ == "__main__":
    main()