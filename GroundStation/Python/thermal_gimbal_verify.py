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
import time

thermal_image_folder = r"C:\development\XPRIZE_Finals\WildBridge\ThermalImages"
raw_image_folder = r"C:\development\XPRIZE_Finals\WildBridge\RawImages"
TEMP_THRESHOLD = float(70)


class Verifier:
    def __init__(self, thermal_img_dir, raw_img_dir, dji_interface):
        self.thermal_image_folder = thermal_img_dir
        self.raw_image_folder = raw_img_dir
        self.drone = dji_interface

    def takeThermalImage(self):
        save_path = os.path.join(self.thermal_image_folder, f"thermal_image_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
        save_path_raw = os.path.join(self.raw_image_folder, f"raw_image_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.raw")

        all_states_resp = self.drone.requestAllStates()
        print("All states response:", all_states_resp)
        t_img_response = self.drone.requestCaptureThermalImage(save_path)
        print("Thermal image capture response:", t_img_response)

        if t_img_response is None:
            print("Failed to send capture request")
            return None, None

        elif t_img_response == "T_IMG_SAVE_FAILURE":
            print("Failed to save thermal image.")
            return None, None

        elif t_img_response == "T_IMG_CAP_FAILURE":
            print("Failed to capture thermal image.")
            return None, None

        elif t_img_response == "T_IMG_SAVE_SUCCESS":
            print("Thermal image saved successfully.")

        return save_path, all_states_resp

    def rotateGimbal(self, pitch, yaw):
        self.drone.requestSendGimbalPitch(pitch)
        print(f"Gimbal pitch set to {pitch} degrees.")
        time.sleep(0.5)
        self.drone.requestSendGimbalYaw(yaw)
        print(f"Gimbal yaw set to {yaw} degrees.")
        time.sleep(0.5)

    def convert2raw(self, thermal_img_list):
        raw_image_paths = []

        for thermal_img_path in thermal_img_list:
            save_path_raw = os.path.join(self.raw_image_folder, f"raw_image_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.raw")
            try:
                subprocess.run([
                    r"C:\Users\Aditya Shrikhande\Downloads\dji_thermal_sdk_v1.7_20241205\utility\bin\windows\release_x64\dji_irp.exe",
                    "-s", thermal_img_path,
                    "-a", "measure",
                    "--measurefmt", "1",
                    "-o", save_path_raw
                ], check=True)
                print(f"Converted {thermal_img_path} to raw format successfully.")
                raw_image_paths.append(save_path_raw)

            except subprocess.CalledProcessError as e:
                print(f"Failed to convert {thermal_img_path} to raw format. Error: {e}")

        return raw_image_paths

    def analyseImage(self, thermal_img_path, raw_img_path, all_states_response):
        thermal_image = cv2.imread(thermal_img_path, cv2.IMREAD_UNCHANGED)
        temperature_array = np.fromfile(raw_img_path, dtype=np.int16).astype(np.float32) / 10.0
        temperature_array = temperature_array.reshape((thermal_image.shape[0], thermal_image.shape[1]))
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
                if cv2.contourArea(contour) > 0.01:
                    x, y, w, h = cv2.boundingRect(contour)
                    thermal_image = cv2.rectangle(thermal_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    pos={}
                    pos["x"] = (x + x+w)*0.5
                    pos["y"] = y
                    pos_list.append(pos)

                else:
                    continue
        else:
            print("No contours found in the thermal image.")

        warm_bin = np.where((temperature_array >= 30) & (temperature_array <= 40), 255, 0).astype(np.uint8)
        warm_contours, _ = cv2.findContours(warm_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if warm_contours is not None:
            for contour in warm_contours:
                if cv2.contourArea(contour) > 0.01:
                    x, y, w, h = cv2.boundingRect(contour)
                    thermal_image = cv2.rectangle(thermal_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

        self.gps_pos_list = []

        if all_states_response == {}:
            print("Failed to retrieve all states.")
            return

        loc = all_states_response.get("location", {})
        att = all_states_response.get("attitude", {})
        lat = loc.get("latitude", None)
        lon = loc.get("longitude", None)
        alt = loc.get("altitude", None)
        pitch = att.get("pitch", None)
        roll = att.get("roll", None)
        yaw = att.get("yaw", None)
        gimb_att = all_states_response.get("gimbalAttitude", {})
        gimb_pitch = gimb_att.get("pitch", None)
        gimb_roll = gimb_att.get("roll", None)
        gimb_yaw = gimb_att.get("yaw", None)

        print(f"Lat: {lat}, Lon: {lon}, Alt: {alt}\nPitch: {pitch}, Roll: {roll}, Yaw: {yaw}\nGimbPitch: {gimb_pitch}, GimbRoll: {gimb_roll}, GimbYaw: {gimb_yaw}")

        # for pos in pos_list:
        #     gps_pos = {}
        #     temp = self.localiser.getObjectPosition_flatPixel(pos["x"], pos["y"], DJI_M4T["F_thermal"], DJI_M4T["cx_thermal"], DJI_M4T["cy_thermal"], gimb_pitch, gimb_yaw, lat, lon, alt)
        #     gps_pos["Lat"] = temp["FireLat"]
        #     gps_pos["Lon"] = temp["FireLon"]
        #     self.gps_pos_list.append(gps_pos)

        # print("GPS Positions:", self.gps_pos_list)
        return thermal_image


if __name__ == "__main__":
    ip_rc = "172.20.10.2"
    dji_interface = DJIInterfaceLite(ip_rc)
    verifier = Verifier(thermal_image_folder, raw_image_folder, dji_interface)
    print("Drone connected.")

    thermal_save_paths = []
    all_states_responses = []

    while True:
        usr_input = input("\n[t] Take thermal image | [g] Rotate gimbal | [q] Quit: ").strip().lower()

        if usr_input == "q":
            break

        elif usr_input == "t":
            result = verifier.takeThermalImage()
            if result[0] is not None:
                thermal_save_paths.append(result[0])
                all_states_responses.append(result[1])

        elif usr_input == "g":
            gimbal_input = input("Enter gimbal (pitch, yaw) e.g. -30,45: ").strip()
            try:
                parts = gimbal_input.split(",")
                pitch_angle = float(parts[0].strip())
                yaw_angle = float(parts[1].strip())
                verifier.rotateGimbal(pitch_angle, yaw_angle)
            except (ValueError, IndexError):
                print("Invalid input. Use format: pitch,yaw (e.g. -30,45)")

        else:
            print("Unknown command.")

    if not thermal_save_paths:
        print("No thermal images captured. Exiting.")
        sys.exit(0)

    print(f"\nConverting {len(thermal_save_paths)} thermal images to raw...")
    raw_image_paths = verifier.convert2raw(thermal_save_paths)

    print(f"\nAnalysing {len(raw_image_paths)} images...")
    for idx, raw_img in enumerate(raw_image_paths):
        thermal_img_path = thermal_save_paths[idx]
        all_states_response = all_states_responses[idx]
        analysed_image = verifier.analyseImage(thermal_img_path, raw_img, all_states_response)
        if analysed_image is not None:
            cv2.imshow("Analysed Thermal Image", analysed_image)
            cv2.waitKey(0)

    cv2.destroyAllWindows()
