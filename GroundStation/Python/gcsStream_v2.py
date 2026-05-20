import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QGridLayout, QLabel, QVBoxLayout
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from viewerObjects import MultiStreamViewer
import cv2
import numpy as np
from djiInterfaceLite import DJIInterfaceLite
from objectPosition import ObjectPosition
from thermalImageStream import ThermalImageAnalyser
import time
import threading
import queue
from typing import Optional, List
from ultralytics import YOLO
from tracker import SimpleTracker
from analysisVars import AnalysisVars
from vehicleParameters import DJI_M300RTK_A, DJI_M300RTK_B, DJI_M300RTK_C, DJI_M300RTK_D, DJI_M300RTK_E, DJI_M4T, DJI_M3E, DJI_M400

# --- MODEL CONFIGURATION ---

model_path = r"C:\development\NVIDIA_brev_files\train_results\final_weights\train7\weights\best.pt"  # <-- Set your YOLO model path here
confidence_threshold = 0.25
nms_iou_threshold = 0.45

# Detection and tracking configuration
DETECTION_INTERVAL = 5  # Run detection every N frames
TRACKING_ENABLED = True  # Enable tracking between detections

class_names = {
    0: "smoke",
    1: "fire"
}


class InferenceWorker(threading.Thread):
    def __init__(self, model_path: str):
        super().__init__(daemon=True)
        self.queue = queue.Queue()
        self.model = YOLO(model_path, verbose=False)
        self.model.conf = confidence_threshold
        self.model.iou = nms_iou_threshold
        self.running = True

    def run(self):
        while self.running:
            try:
                frame, result_event, result_holder = self.queue.get(timeout=0.5)
            except queue.Empty:
                continue
            results = self.model(frame, verbose=False)[0]
            detections = []
            for box in results.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                if cls_id in class_names and conf >= confidence_threshold:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'class': cls_id,
                        'confidence': conf
                    })
            result_holder.append(detections)
            result_event.set()

    def submit(self, frame):
        result_event = threading.Event()
        result_holder = []
        self.queue.put((frame, result_event, result_holder))
        return result_event, result_holder

    def stop(self):
        self.running = False


class ProcessedStreamThread(QThread):
    frame_ready = pyqtSignal(QPixmap, int)
    stream_status = pyqtSignal(int, bool, str)

    def __init__(self, stream_url: str, stream_id: int, inference_worker: InferenceWorker, drone_interface: DJIInterfaceLite, drone: dict):
        super().__init__()
        self.stream_url = stream_url
        self.stream_id = stream_id
        self.running = True
        self.drone_interface = drone_interface
        self.drone = drone
        self.inference_worker = inference_worker
        self.pending_result = None
        self.pending_event = None
        self.localiser = ObjectPosition(USE_ELEVATION=False)
        self.thermal_analyser = ThermalImageAnalyser(self.drone_interface, self.localiser)
        self.global_vars = AnalysisVars()
        
        # Initialize tracker
        self.tracker = SimpleTracker(max_disappeared=10, max_distance=100)
        self.detection_frame_counter = stream_id % DETECTION_INTERVAL
        
        # Reliability settings
        self.reconnect_timeout = 5  # seconds between reconnection attempts
        self.read_timeout = 3.0     # seconds timeout for frame reading
        self.frozen_threshold = 5.0  # seconds to detect frozen stream
        self.last_frame_time = time.time()
        self.consecutive_failures = 0
        self.max_failures = 3
        self.cap = None
        
    def connect_to_stream(self) -> bool:
        """Attempt to connect to RTSP stream with proper error handling"""
        try:
            if self.cap is not None:
                self.cap.release()
            
            self.cap = cv2.VideoCapture(self.stream_url, cv2.CAP_FFMPEG)
            # CRITICAL: Set buffer to 1 to minimize latency
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            # Use lower resolution for faster processing
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            # Set FPS
            self.cap.set(cv2.CAP_PROP_FPS, 15)
            # Reduce codec delay
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H', '2', '6', '4'))
            
            if not self.cap.isOpened():
                self.stream_status.emit(self.stream_id, False, "Failed to open stream")
                return False
            
            # Test read to verify connection
            ret, frame = self.cap.read()
            if not ret or frame is None:
                self.stream_status.emit(self.stream_id, False, "Failed to read initial frame")
                return False
            
            self.stream_status.emit(self.stream_id, True, "Connected successfully")
            self.consecutive_failures = 0
            return True
            
        except Exception as e:
            self.stream_status.emit(self.stream_id, False, f"Connection error: {str(e)}")
            return False

    def read_frame_with_timeout(self):
        if self.cap is None:
            return False, None

        try:
            # Drop stale buffered frames, but don't fail on partial grabs
            for _ in range(2):
                if not self.cap.grab():
                    break

            ret, frame = self.cap.read()

            if ret and frame is not None:
                self.last_frame_time = time.time()
                return True, frame
            else:
                return False, None

        except Exception as e:
            self.stream_status.emit(self.stream_id, False, f"Frame read error: {str(e)}")
            return False, None
        
    def run(self):
        i = 0
        img_coords_list = []
        self.gps_pos_list = []

        # Main reconnection loop
        while self.running:
            try:
                # Attempt to connect to stream
                if not self.connect_to_stream():
                    self.consecutive_failures += 1
                    if self.consecutive_failures >= self.max_failures:
                        self.stream_status.emit(self.stream_id, False, f"Max failures reached ({self.max_failures})")
                        time.sleep(self.reconnect_timeout * 2)  # Longer wait after max failures
                    else:
                        time.sleep(self.reconnect_timeout)
                    continue

                # Stream processing loop
                while self.running:
                    start_time = time.time()
                    ret, frame = self.read_frame_with_timeout()
                    
                    if not ret or frame is None:
                        self.consecutive_failures += 1
                        if self.consecutive_failures >= self.max_failures:
                            self.stream_status.emit(self.stream_id, False, "Frame read failed, attempting reconnection")
                            break
                        time.sleep(0.05)
                        continue

                    # Reset failure counter on successful frame
                    self.consecutive_failures = 0

                    # Downscale for faster processing and display
                    frame = cv2.resize(frame, (640, 480))

                    # CRITICAL: Skip frames for low latency - only process every frame
                    if i % 1 != 0:
                        i += 1
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        h, w, ch = rgb.shape
                        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
                        self.frame_ready.emit(QPixmap.fromImage(qimg), self.stream_id)
                        continue

                    # Check if pending inference result is ready
                    got_detection = False
                    if self.pending_event is not None and self.pending_event.is_set():
                        current_detections = self.pending_result[0]
                        self.pending_event = None
                        self.pending_result = None
                        tracked_objects = self.tracker.update(current_detections)
                        got_detection = True
                        print(f"Stream {self.stream_id}: Detection result - {len(current_detections)} detections, tracking {len(tracked_objects)} objects")
                    else:
                        tracked_objects = self.tracker.update()

                    # Submit new detection request if interval reached and no pending request
                    run_detection = (self.detection_frame_counter % DETECTION_INTERVAL == 0) or not TRACKING_ENABLED
                    if run_detection and self.pending_event is None:
                        self.pending_event, self.pending_result = self.inference_worker.submit(frame.copy())

                    # Get current predictions (either from detection or tracking)
                    predictions_to_draw = self.tracker.get_predictions()

                    # Draw all tracked objects
                    for pred in predictions_to_draw:
                        x1, y1, x2, y2 = map(int, pred['bbox'])
                        label = pred['class']
                        score = pred['confidence']
                        track_id = pred.get('track_id', -1)
                        
                        if label in class_names:
                            label_name = class_names[label]
                            color = (0, 0, 255) if label_name == "fire" else (255, 165, 0)  # Red for fire, orange for smoke

                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                            
                            # Show detection vs tracking status
                            status = "DET" if got_detection else "TRK"
                            label_text = f"{label_name} {score:.2f} [{status}:{track_id}]"
                            cv2.putText(frame, label_text, (x1, y1 - 8),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 3)

                        # Only do GPS calculations every 60 frames (was 30) and only on detection frames
                        if (i == 0 or i % 60 == 0) and got_detection:
                            img_coords_list.clear()
                            self.gps_pos_list.clear()
                            
                            # Move network requests to separate thread to avoid blocking
                            def update_gps_async():
                                try:
                                    all_states = self.drone_interface.requestAllStates()
                                    att = all_states.get("attitude", {})
                                    loc = all_states.get("location", {})
                                    gb_att = all_states.get("gimbalAttitude", {})
                                    lat = loc.get("latitude", None)
                                    lon = loc.get("longitude", None)
                                    alt = loc.get("altitude", None)
                                    roll = att.get("roll", None)
                                    pitch = att.get("pitch", None)
                                    yaw = att.get("yaw", None)
                                    gb_pitch = gb_att.get("pitch", None)
                                    gb_yaw = gb_att.get("yaw", None)
                                    
                                    # print(all_states)
                                    return lat, lon, alt, roll, pitch, yaw, gb_pitch, gb_yaw
                                except:
                                    return None, None, None, None, None
                            
                            # Use detection results for GPS calculation
                            for pred in predictions_to_draw:
                                x1, y1, x2, y2 = pred['bbox']
                                x_c = x1 + (x2 - x1) / 2
                                y_c = y1 + (y2 - y1) / 2
                                label = pred['class']
                                score = pred['confidence']
                                
                                if label in class_names:
                                    class_name = class_names[label]
                                    img_coords_dict = {
                                        "obj": label,
                                        "class_name": class_name,
                                        "confidence": score,
                                        "x_c": x_c,
                                        "y_c": y_c,
                                        "box": [x1, y1, x2, y2]
                                    }
                                    img_coords_list.append(img_coords_dict)

                            lat, lon, alt, roll, pitch, yaw, gb_pitch, gb_yaw = update_gps_async()

                            for pos in img_coords_list:
                                # Initialize gps_info with default values
                                gps_info = {"Obj": pos['obj'], "Lat": "N/A", "Lon": "N/A"}
                                
                                if None not in (lat, lon, alt, gb_pitch, gb_yaw):
                                    try:
                                        gps_pos = self.localiser.getObjectPosition(
                                            pos["x_c"], pos["y_c"],
                                            f=self.drone['F_video'],
                                            cx=self.drone['cx_video'],
                                            cy=self.drone['cy_video'],
                                            gb_pitch=gb_pitch,
                                            gb_yaw=0,
                                            lat_drone=lat,
                                            lon_drone=lon,
                                            alt_drone=alt
                                        )
                                        gps_info = {"Obj": pos['obj'], "Lat": gps_pos['FireLat'], "Lon": gps_pos['FireLon']}
                                        
                                        # Send fire and/or smoke location in background (non-blocking)
                                        if gps_info['Obj'] == 1:  # Smoke
                                            threading.Thread(target=lambda: self.drone_interface.requestSendSmokeLocation(gps_pos['FireLat'], gps_pos['FireLon']), daemon=True).start()

                                        elif gps_info['Obj'] == 2:  # Fire
                                            threading.Thread(target=lambda: self.drone_interface.requestSendFireLocation(gps_pos['FireLat'], gps_pos['FireLon']), daemon=True).start()

                                    except Exception as e:
                                        print(f"GPS calculation error: {e}")
                                        # gps_info already has default values, no need to access undefined variables
                                
                                # Always append gps_info (either calculated or default)
                                self.gps_pos_list.append(gps_info)
                                self.global_vars.visual_fireLoc = [g for g in self.gps_pos_list if g['Obj'] == 1]
                                self.global_vars.visual_smokeLoc = [g for g in self.gps_pos_list if g['Obj'] == 2]
                                # threading.Thread(target=lambda: self.thermal_analyser.confirm(), daemon=True).start()

                    i += 1
                    self.detection_frame_counter += 1
                    
                    # Convert to QPixmap in worker thread to offload main thread
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb.shape
                    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
                    pixmap = QPixmap.fromImage(qimg)
                    self.frame_ready.emit(pixmap, self.stream_id)
                    
                    process_time = time.time() - start_time
                    target_fps = 10
                    frame_time = 1.0 / target_fps
                    if process_time < frame_time:
                        time.sleep(frame_time - process_time)

            except Exception as e:
                self.stream_status.emit(self.stream_id, False, f"Unexpected error: {str(e)}")
                print(f"Stream {self.stream_id} error: {str(e)}")
                time.sleep(self.reconnect_timeout)

        # Cleanup
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def stop(self):
        self.running = False

class ProcessedMultiStreamViewer(MultiStreamViewer):
    def __init__(self, num_drones, model_path: str):
        super().__init__(num_drones)
        self.setWindowTitle("Processed Multi-Stream Viewer")
        self.stream_status = {}  # Track stream health
        print("Loading YOLO model...")
        self.inference_worker = InferenceWorker(model_path)
        self.inference_worker.start()
        print("Model loaded.")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Q:
            print("Exiting...")
            self.close()

    def closeEvent(self, event):
        self.inference_worker.stop()
        for thread in self.stream_threads.values():
            thread.running = False
        for thread in self.stream_threads.values():
            thread.wait(3000)
            if thread.isRunning():
                thread.terminate()
        event.accept()
        QApplication.instance().quit()

    def update_stream(self, pixmap, stream_id):
        if stream_id in self.stream_windows:
            window = self.stream_windows[stream_id]
            scaled = pixmap.scaled(window.size(), Qt.AspectRatioMode.KeepAspectRatio)
            window.label.setPixmap(scaled)

    def handle_stream_status(self, stream_id: int, is_alive: bool, message: str):
        """Handle stream status updates"""
        self.stream_status[stream_id] = is_alive
        status_text = "LIVE" if is_alive else "DISCONNECTED"
        print(f"Stream {stream_id}: {status_text} - {message}")
        
        # Optionally update window title or add visual indicators
        if stream_id in self.stream_windows:
            window = self.stream_windows[stream_id]
            if not is_alive:
                # You could add a red border or overlay here to indicate disconnection
                window.setStyleSheet("border: 3px solid red;")
            else:
                window.setStyleSheet("border: 3px solid green;")

    def add_stream(self, stream_url: str, stream_id: int, drone_name: str) -> bool:
        """Add a new processed stream to the viewer"""
        if stream_id < 0 or stream_id >= self.num_drones:
            return False
        
        if drone_name == "M300RTK_A":
            self.drone_interface = DJIInterfaceLite(DJI_M300RTK_A["IP_RC"])
            self.drone = DJI_M300RTK_A
        elif drone_name == "M300RTK_B":
            self.drone_interface = DJIInterfaceLite(DJI_M300RTK_B["IP_RC"])
            self.drone = DJI_M300RTK_B
        elif drone_name == "M300RTK_C":
            self.drone_interface = DJIInterfaceLite(DJI_M300RTK_C["IP_RC"])
            self.drone = DJI_M300RTK_C
        elif drone_name == "M300RTK_D":
            self.drone_interface = DJIInterfaceLite(DJI_M300RTK_D["IP_RC"])
            self.drone = DJI_M300RTK_D
        elif drone_name == "M300RTK_E":
            self.drone_interface = DJIInterfaceLite(DJI_M300RTK_E["IP_RC"])
            self.drone = DJI_M300RTK_E
        elif drone_name == "M4T":
            self.drone_interface = DJIInterfaceLite(DJI_M4T["IP_RC"])
            self.drone = DJI_M4T
        elif drone_name == "M3E":
            self.drone_interface = DJIInterfaceLite(DJI_M3E["IP_RC"])
            self.drone = DJI_M3E
        elif drone_name == "M400":
            self.drone_interface = DJIInterfaceLite(DJI_M400["IP_RC"])
            self.drone = DJI_M400
        else:
            print(f"Unknown drone name: {drone_name}")
            return False
        
        # Create and start processed stream thread
        thread = ProcessedStreamThread(stream_url, stream_id, self.inference_worker, self.drone_interface, self.drone)
        thread.frame_ready.connect(self.update_stream)
        thread.stream_status.connect(self.handle_stream_status)  # Connect status handler
        self.stream_threads[stream_id] = thread
        thread.start()
        return True

def main():
    app = QApplication(sys.argv) 

    ############# Set number of drones here #############
    num_drones = 5
    ####################################################
    
    # Create viewer with model
    viewer = ProcessedMultiStreamViewer(num_drones, model_path)
    viewer.show()

    # RTSP stream URLs
    # stream_urls = {"M4T": f"rtsp://aaa:aaa@{DJI_M4T['IP_RC']}:8554/streaming/live/1",
    #                "M300RTK_A": f"rtsp://aaa:aaa@{DJI_M300RTK_A['IP_RC']}:8554/streaming/live/1",
    #                "M300RTK_B": f"rtsp://aaa:aaa@{DJI_M300RTK_B['IP_RC']}:8554/streaming/live/1"}
    
    video_sources = {"M300RTK_A": r"C:\Users\Aditya Shrikhande\Downloads\DJI_20220721141932_0008_W.MP4",
                     "M300RTK_B": r"C:\Users\Aditya Shrikhande\Downloads\DJI_20220721141735_0007_W.MP4",
                     "M300RTK_C": r"C:\Users\Aditya Shrikhande\Downloads\DJI_20220721141419_0006_W.MP4",
                     "M300RTK_D": r"C:\Users\Aditya Shrikhande\Downloads\DJI_20220721141130_0005_W.MP4",
                     "M300RTK_E": r"C:\Users\Aditya Shrikhande\Downloads\DJI_20220721141011_0004_W.MP4"
                     }
    
    # video_sources = {"M300RTK_A": r"C:\Users\Aditya Shrikhande\Downloads\DJI_20220721141735_0007_W.MP4"}
    # # Add streams to viewer
    
    # for i, (drone_name, url) in enumerate(stream_urls.items()):
    #     viewer.add_stream(url, i, drone_name)

    for i, (drone_name, source) in enumerate(video_sources.items()):
        viewer.add_stream(source, i, drone_name)

    sys.exit(app.exec())

if __name__ == '__main__':
    main()