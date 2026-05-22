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

def main():
    ip_rc = "172.20.10.2"
    dji_interface = DJIInterfaceLite(ip_rc)
    print("Drone connected...")
    
    usr_input = None

    while usr_input != "q":
        usr_input = input("Enter 'p' to set gimbal pitch, 'y' to set gimbal yaw, or 'q' to quit: ")

        if usr_input == "p":
            pitch_input = input("Enter desired gimbal pitch angle (in degrees): ")
            try:
                pitch_angle = float(pitch_input)
                dji_interface.requestSendGimbalPitch(pitch_angle)
                print(f"Gimbal pitch set to {pitch_angle} degrees...")
            except ValueError:
                print("Invalid input. Please enter a numeric value for the pitch angle.")

            time.sleep(1)
        elif usr_input == "y":
            yaw_input = input("Enter desired gimbal yaw angle (in degrees): ")
            try:
                yaw_angle = float(yaw_input)
                dji_interface.requestSendGimbalYaw(yaw_angle)
                print(f"Gimbal yaw set to {yaw_angle} degrees...")
            except ValueError:
                print("Invalid input. Please enter a numeric value for the yaw angle.")

            time.sleep(1)

    return

if __name__ == "__main__":
    main()