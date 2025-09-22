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
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
from torchvision import transforms
import time
import threading
import torch
from typing import Optional, List
from ultralytics import YOLO
from tracker import SimpleTracker
from analysisVars import AnalysisVars
from vehicleParameters import DJI_M300RTK_A, DJI_M300RTK_B, DJI_M4T, DJI_M3E, DJI_M400

# --- MODEL CONFIGURATION ---

num_classes = 3                   # background + fire + smoke
confidence_threshold = 0.99999
nms_iou_threshold = 0.3          # IoU threshold for NMS
model_path = r"C:\development\ULTRA Code v2\Faster_RCNN\fasterrcnn_yolo_trained.pth"

# Detection and tracking configuration
DETECTION_INTERVAL = 5  # Run detection every N frames
TRACKING_ENABLED = True  # Enable tracking between detections

class_names = {
    1: "smoke",
    2: "fire"
}

# --- DEVICE ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = fasterrcnn_resnet50_fpn(pretrained=False)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()


class ProcessedStreamThread(QThread):
    frame_ready = pyqtSignal(np.ndarray, int)
    stream_status = pyqtSignal(int, bool, str)  # stream_id, is_alive, status_message

    def __init__(self, stream_url: str, stream_id: int, model_path: str,  drone_interface: DJIInterfaceLite, drone: dict):
        super().__init__()
        self.stream_url = stream_url
        self.stream_id = stream_id
        self.running = True
        self.drone_interface = drone_interface
        self.drone = drone

        # Load YOLOv8 model with verbose=False to suppress stdout messages
        self.model = YOLO(model_path, verbose=False)
        self.model.conf = 0.25  # confidence threshold
        self.model.iou = 0.45   # NMS IOU threshold
        self.localiser = ObjectPosition(USE_ELEVATION=False)
        self.thermal_analyser = ThermalImageAnalyser(self.drone_interface, self.localiser)
        self.global_vars = AnalysisVars()
        
        # Initialize tracker
        self.tracker = SimpleTracker(max_disappeared=10, max_distance=100)
        self.detection_frame_counter = 0
        
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
        """Read frame with timeout and frame dropping for low latency"""
        if self.cap is None:
            return False, None
            
        try:
            # Drop old frames by reading multiple times
            frame = None
            for _ in range(3):  # Try to get the latest frame
                ret = self.cap.grab()
                if not ret:
                    return False, None
            
            # Retrieve the latest frame
            ret, frame = self.cap.retrieve()
            
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
                        self.stream_status.emit(self.stream_id, False, "Frame read failed, attempting reconnection")
                        break

                    # Reset failure counter on successful frame
                    self.consecutive_failures = 0

                    # CRITICAL: Skip frames for low latency - only process every frame
                    if i % 1 != 0:
                        i += 1
                        # Still emit frame for display but without detection overlay
                        self.frame_ready.emit(frame, self.stream_id)
                        continue

                    # Determine if we should run detection or just tracking
                    run_detection = (self.detection_frame_counter % DETECTION_INTERVAL == 0) or not TRACKING_ENABLED
                    
                    current_detections = []
                    
                    if run_detection:
                        # Run Faster R-CNN detection
                        with torch.no_grad():
                            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            pil_img = Image.fromarray(rgb_frame)
                            img_tensor = transforms.ToTensor()(pil_img).to(device)
                            predictions = model([img_tensor])[0]

                            # Apply confidence threshold
                            keep_conf = predictions['scores'] >= confidence_threshold
                            boxes = predictions['boxes'][keep_conf]
                            scores = predictions['scores'][keep_conf]
                            labels = predictions['labels'][keep_conf]

                            # Perform NMS per class
                            final_boxes = []
                            final_scores = []
                            final_labels = []

                            for class_id in class_names.keys():
                                # Get detections for this class
                                class_mask = labels == class_id
                                class_boxes = boxes[class_mask]
                                class_scores = scores[class_mask]

                                if len(class_boxes) > 0:
                                    # Apply NMS
                                    keep_indices = torch.ops.torchvision.nms(
                                        class_boxes,
                                        class_scores,
                                        nms_iou_threshold
                                    )

                                    # Keep the selected detections
                                    final_boxes.extend(class_boxes[keep_indices])
                                    final_scores.extend(class_scores[keep_indices])
                                    final_labels.extend([class_id] * len(keep_indices))

                            # Convert to detection format for tracker
                            for box, score, label in zip(final_boxes, final_scores, final_labels):
                                if label in class_names:
                                    x1, y1, x2, y2 = map(int, box.tolist())
                                    current_detections.append({
                                        'bbox': [x1, y1, x2, y2],
                                        'class': label,
                                        'confidence': score
                                    })

                        # Update tracker with new detections
                        tracked_objects = self.tracker.update(current_detections)
                        print(f"Stream {self.stream_id}: Detection frame - Found {len(current_detections)} detections, tracking {len(tracked_objects)} objects")
                    
                    else:
                        # Only tracking - no new detections
                        tracked_objects = self.tracker.update()
                        print(f"Stream {self.stream_id}: Tracking frame - {len(tracked_objects)} objects tracked")

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
                            status = "DET" if run_detection else "TRK"
                            label_text = f"{label_name} {score:.2f} [{status}:{track_id}]"
                            cv2.putText(frame, label_text, (x1, y1 - 8),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 3)

                        # Only do GPS calculations every 60 frames (was 30) and only on detection frames
                        if (i == 0 or i % 60 == 0) and run_detection:
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
                    
                    # Emit the processed frame
                    self.frame_ready.emit(frame, self.stream_id)
                    
                    # Maintain reasonable frame rate without using cv2.waitKey
                    process_time = time.time() - start_time
                    target_fps = 30
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
        if self.cap is not None:
            self.cap.release()
            self.cap = None

class ProcessedMultiStreamViewer(MultiStreamViewer):
    def __init__(self, num_drones, model_path: str):
        super().__init__(num_drones)
        self.model_path = model_path
        self.setWindowTitle("Processed Multi-Stream Viewer")
        self.stream_status = {}  # Track stream health

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
        thread = ProcessedStreamThread(stream_url, stream_id, self.model_path, self.drone_interface, self.drone)
        thread.frame_ready.connect(self.update_stream)
        thread.stream_status.connect(self.handle_stream_status)  # Connect status handler
        self.stream_threads[stream_id] = thread
        thread.start()
        return True

def main():
    app = QApplication(sys.argv) 

    ############# Set number of drones here #############
    num_drones = 3
    ####################################################
    
    # Create viewer with model
    viewer = ProcessedMultiStreamViewer(num_drones, model_path)
    viewer.show()

    # RTSP stream URLs
    stream_urls = {"M4T": f"rtsp://aaa:aaa@{DJI_M4T['IP_RC']}:8554/streaming/live/1",
                   "M300RTK_A": f"rtsp://aaa:aaa@{DJI_M300RTK_A['IP_RC']}:8554/streaming/live/1",
                   "M300RTK_B": f"rtsp://aaa:aaa@{DJI_M300RTK_B['IP_RC']}:8554/streaming/live/1"}
    
    # video_sources = {"M300RTK_A": r"C:\Users\Aditya Shrikhande\Downloads\DJI_20220721141735_0007_W.MP4",
    #                  "M300RTK_B": r"C:\development\WildbridgeFireVision_v3\Images\DJI_202508191432_012_IncidentOperations-Waypoint1\DJI_20250819151016_0009_W.MP4"
    #                  }
    # Add streams to viewer
    
    for i, (drone_name, url) in enumerate(stream_urls.items()):
        viewer.add_stream(url, i, drone_name)

    # for i, (drone_name, source) in enumerate(video_sources.items()):
    #     viewer.add_stream(source, i, drone_name)

    sys.exit(app.exec())

if __name__ == '__main__':
    main()