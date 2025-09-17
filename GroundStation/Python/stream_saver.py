import cv2
import threading
import time
import os
from datetime import datetime

class StreamSaver:
    def __init__(self, rtsp_url, output_dir="recordings", duration=30):
        self.rtsp_url = rtsp_url
        self.output_dir = output_dir
        self.duration = duration
        self.is_recording = False
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def start_recording(self):
        """Start recording a 30-second video clip"""
        if self.is_recording:
            print("Already recording...")
            return
        
        self.is_recording = True
        threading.Thread(target=self._record_video, daemon=True).start()
    
    def _record_video(self):
        """Record video for specified duration"""
        try:
            # Connect to RTSP stream
            cap = cv2.VideoCapture(self.rtsp_url)
            
            if not cap.isOpened():
                print(f"Error: Could not connect to RTSP stream {self.rtsp_url}")
                self.is_recording = False
                return
            
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Create output filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.output_dir, f"recording_{timestamp}.mp4")
            
            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
            
            print(f"Recording started: {output_file}")
            start_time = time.time()
            
            while time.time() - start_time < self.duration:
                ret, frame = cap.read()
                if not ret:
                    print("Error reading frame from stream")
                    break
                
                # cv2.imshow("Recording...", frame)
                out.write(frame)
            
            print(f"Recording completed: {output_file}")
            
        except Exception as e:
            print(f"Error during recording: {e}")
        
        finally:
            cap.release()
            out.release()
            self.is_recording = False

def main():
    # Example usage
    # rtsp_url = "rtsp://aaa:aaa@192.168.51.142:8554/streaming/live/1"  # Replace with your RTSP URL
    rtsp_url = "rtsp://aaa:aaa@192.168.51.197:8554/streaming/live/1"
    # rtsp_url = "rtsp://aaa:aaa@192.168.51.67:8554/streaming/live/1"
    output_dir = r"C:\development\WildbridgeFireVision_v3\WildBridge\Recordings\M4T"
    duration = 30  # seconds
    
    saver = StreamSaver(rtsp_url, output_dir, duration)
    
    print("Press 'r' to start recording, 'q' to quit")
    while True:
        command = input().strip().lower()
        if command == 'r':
            saver.start_recording()
        elif command == 'q':
            break
        else:
            print("Invalid command. Use 'r' to record or 'q' to quit")

if __name__ == "__main__":
    main()