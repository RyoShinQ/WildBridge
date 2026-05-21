import sys
import numpy as np
import cv2
import datetime
from djiInterfaceLite import DJIInterfaceLite
from objectPosition import ObjectPosition
from analysisVars import AnalysisVars
import subprocess
import os
from geopy.distance import geodesic
from vehicleParameters import DJI_M4T


# Environment variables
TEMP_THRESHOLD = float(30)  # Temperature threshold in Celsius
ZENMUSE_WIDTH = 7.68
ZENMUSE_HEIGHT = 6.14
ZENMUSE_FOCAL_LENGTH = 13.5
ZENMUSE_FX = ZENMUSE_FOCAL_LENGTH*(640/ZENMUSE_WIDTH)
ZENMUSE_FY = ZENMUSE_FOCAL_LENGTH*(512/ZENMUSE_HEIGHT)
ZENMUSE_CX = 320
ZENMUSE_CY = 256


class ThermalImageAnalyser:
    def __init__(self, dji_interface, object_localiser):
        self.dji = dji_interface
        self.localiser = object_localiser
        self.thermal_image_folder = r"C:\development\WildbridgeFireVisionV2\WildBridge\ThermalImages"
        self.raw_image_folder = r"C:\development\WildbridgeFireVisionV2\WildBridge\RawImages"
        self.global_vars = AnalysisVars()

    def requestImage(self):
        self.save_path = os.path.join(self.thermal_image_folder, f"thermal_image_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
        self.save_path_raw = os.path.join(self.raw_image_folder, f"raw_image_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.raw")

        self.all_states = self.dji.requestAllStates()
        t_img_response = self.dji.requestCaptureThermalImage(self.save_path)

        if t_img_response is None:
            print("Failed to send capture request")
        
        elif t_img_response == "T_IMG_SAVE_FAILURE":
            print("Failed to save thermal image.")
        
        elif t_img_response == "T_IMG_CAP_FAILURE":
            print("Failed to capture thermal image.")
    
        elif t_img_response == "T_IMG_SAVE_SUCCESS":
            print("Thermal image saved successfully.")
        
        try:
            subprocess.run([
                r"C:\Users\Aditya Shrikhande\Downloads\dji_thermal_sdk_v1.7_20241205\utility\bin\windows\release_x64\dji_irp.exe",
                "-s", self.save_path,
                "-a", "measure",
                "-o", self.save_path_raw,
                "--measurefmt", "float32"
            ], check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running DJI IRP utility: {e}")
            print(f"Error output: {e.stderr}")
            sys.exit(1)
        except FileNotFoundError:
            print("DJI IRP utility executable not found")
            sys.exit(1)

        if not os.path.exists(self.save_path_raw):
            print("Temperature array file not found.")
            return
        
        else:
            print("Temperature array file found.")
            return self.save_path, self.save_path_raw

    def analyseImage(self):
        thermal_image = cv2.imread(self.save_path, cv2.IMREAD_UNCHANGED)
        temperature_array = np.fromfile(self.save_path_raw, dtype=np.float32)
        temperature_array = temperature_array.reshape((thermal_image.shape[0]//2, thermal_image.shape[1]//2))
        print("Temperature array shape:", temperature_array.shape)
        print("Temperature array data type:", temperature_array.dtype)
        print("Temperature array min value:", np.min(temperature_array))
        print("Temperature array max value:", np.max(temperature_array))
        temperature_array_bin = np.where(temperature_array > TEMP_THRESHOLD, 255, 0)
        temperature_array_bin = temperature_array_bin.astype(np.uint8)

        contours, _ = cv2.findContours(temperature_array_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        pos_list = []

        if contours is not None:
            for contour in contours:
                if cv2.contourArea(contour) > 100:
                    x, y, w, h = cv2.boundingRect(contour)
                    thermal_image = cv2.rectangle(thermal_image, (2*x, 2*y), (2*x + 2*w, 2*y + 2*h), (0, 255, 0), 2)
                    pos={}
                    pos["x"] = (x + x+w)*0.5
                    pos["y"] = y
                    pos_list.append(pos)
                    
                else:
                    continue
        else:
            print("No contours found in the thermal image.")

        self.gps_pos_list = []


        if self.all_states == {}:
            print("Failed to retrieve all states.")
            return

        loc = self.all_states.get("location", {})
        att = self.all_states.get("attitude", {})
        lat = loc.get("latitude", None)
        lon = loc.get("longitude", None)
        alt = loc.get("altitude", None)
        pitch = att.get("pitch", None)
        roll = att.get("roll", None)
        yaw = att.get("yaw", None)
        gimb_att = self.all_states.get("gimbalAttitude", {})
        gimb_pitch = gimb_att.get("pitch", None)
        gimb_roll = gimb_att.get("roll", None)
        gimb_yaw = gimb_att.get("yaw", None)

        print(f"Lat: {lat}, Lon: {lon}, Alt: {alt}\nPitch: {pitch}, Roll: {roll}, Yaw: {yaw}\nGimbPitch: {gimb_pitch}, GimbRoll: {gimb_roll}, GimbYaw: {gimb_yaw}")

        for pos in pos_list:
            gps_pos = {}
            temp = self.localiser.getObjectPosition_flatPixel(pos["x"], pos["y"], DJI_M4T["F_thermal"], DJI_M4T["cx_thermal"], DJI_M4T["cy_thermal"], gimb_pitch, gimb_yaw, lat, lon, alt)
            gps_pos["Lat"] = temp["FireLat"]
            gps_pos["Lon"] = temp["FireLon"]
            self.gps_pos_list.append(gps_pos)

        print("GPS Positions:", self.gps_pos_list)
        return thermal_image

    def run(self):
        input_command = input("Press 'r' to start the thermal image analysis loop or 'q' to quit: ").strip().lower()
        if input_command == 'q':
            print("Exiting the program.")
            return
        elif input_command != 'r':
            print("Invalid input. Please enter 'r' to run or 'q' to quit.")
            return
        print("Starting the thermal image analysis loop.")
        print("Controls: Press 'a' on the image window to analyse new image, 'q' to quit")
        
        window_shown = False

        while True:
            analyse_flag = input("Press 'a' to request a new thermal image and analyse or 'q' to quit: ").strip().lower()
            if analyse_flag == 'q':
                print("Exiting the program.")
                break
            elif analyse_flag == 'a':
                if window_shown == True:
                    cv2.destroyWindow("Thermal Image Analysis")
                
                self.requestImage()
                thermal_image = self.analyseImage()
                cv2.imshow("Thermal Image Analysis", thermal_image)
                window_shown = True
                
                # Keep the window responsive with a proper event loop
                print("Image displayed. Press 'q' on the image window to continue, or close window.")
                while True:
                    key = cv2.waitKey(30) & 0xFF
                    if key == ord('q'):
                        break
                    # Check if window was closed
                    if cv2.getWindowProperty("Thermal Image Analysis", cv2.WND_PROP_VISIBLE) < 1:
                        break

        cv2.destroyAllWindows()

    def compareLocations(self):
        if not self.global_vars.visual_fireLoc or not self.global_vars.thermal_firelocList:
            print("One or both location lists are empty. Cannot compare.")
            return
        
        confirmed_matches = []

        for v_loc in self.global_vars.visual_fireLoc:
            for t_loc in self.global_vars.thermal_firelocList:
                distance = geodesic((v_loc['Lat'], v_loc['Lon']), (t_loc['Lat'], t_loc['Lon'])).meters
                if distance < 10:  # Threshold distance in meters
                    confirmed_matches.append((v_loc, t_loc))
                    print(f"Confirmed match: Visual {v_loc} with Thermal {t_loc}, Distance: {distance:.2f}m")
        
        if not confirmed_matches:
            print("No confirmed matches found within the threshold distance.")
            return None

        return confirmed_matches

    def confirm(self):
        self.requestImage()
        self.analyseImage()
        self.global_vars.thermal_firelocList = self.gps_pos_list

        confirmed = self.compareLocations()
        if confirmed:
            print(f"Total confirmed matches: {len(confirmed)}")
        else:
            print("No confirmed matches found.")

        self.global_vars.confirmed_matches = confirmed


if __name__ == "__main__":
    ip_rc = "172.20.10.2"
    dji_interface = DJIInterfaceLite(ip_rc)
    object_localiser = ObjectPosition(False)
    thermal_image_analyser = ThermalImageAnalyser(dji_interface, object_localiser)
    thermal_image_analyser.run()