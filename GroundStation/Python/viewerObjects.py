import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QGridLayout, QLabel, QVBoxLayout
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
import numpy as np
import cv2


class VideoStreamThread(QThread):
    frame_ready = pyqtSignal(np.ndarray, int)  # Signal to emit frame and stream ID

    def __init__(self, stream_url, stream_id):
        super().__init__()
        self.stream_url = stream_url
        self.stream_id = stream_id
        self.running = True

    def run(self):
        cap = cv2.VideoCapture(self.stream_url)
        while self.running:
            ret, frame = cap.read()
            if ret:
                self.frame_ready.emit(frame, self.stream_id)
        cap.release()

    def stop(self):
        self.running = False

class StreamWindow(QWidget):
    def __init__(self, stream_id):
        super().__init__()
        self.stream_id = stream_id
        self.layout = QVBoxLayout()
        self.label = QLabel()
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.label)
        self.setLayout(self.layout)
        # Set minimum size for the stream window
        self.setMinimumSize(320, 240)

    def update_frame(self, frame):
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
            self.size(), Qt.AspectRatioMode.KeepAspectRatio)
        self.label.setPixmap(scaled_pixmap)

class MultiStreamViewer(QMainWindow):
    def __init__(self, num_drones=3):
        super().__init__()
        self.setWindowTitle("Multi-Stream Viewer")
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.grid_layout = QGridLayout(self.central_widget)
        self.num_drones = num_drones

        # Create stream windows
        self.stream_windows = {}
        self.stream_threads = {}
        
        # Define grid positions for up to 6 streams (2x3 grid)
        if self.num_drones == 1:
            self.grid_positions = [
                (0, 0)
            ]
        elif self.num_drones == 2:
            self.grid_positions = [
                (0, 0), (0, 1)
            ]
        elif self.num_drones == 3:
            self.grid_positions = [
                (0, 0), (0, 1), (0, 2)
            ]
        elif self.num_drones == 4:
            self.grid_positions = [
                (0, 0), (0, 1),
                (1, 0), (1, 1)
            ]
        elif self.num_drones == 5:
            self.grid_positions = [
                (0, 0), (0, 1), (0, 2),
                (1, 0), (1, 1)
            ]
        elif self.num_drones == 6:
            self.grid_positions = [
                (0, 0), (0, 1), (0, 2),
                (1, 0), (1, 1), (1, 2)
            ]

        # Initialize empty stream windows
        for i in range(num_drones):
            stream_window = StreamWindow(i)
            row, col = self.grid_positions[i]
            self.grid_layout.addWidget(stream_window, row, col)
            self.stream_windows[i] = stream_window

        self.setMinimumSize(1000, 600)

    def add_stream(self, stream_url, stream_id):
        """Add a new stream to the viewer"""
        if stream_id < 0 or stream_id >= 6:
            return False
        
        # Create and start stream thread
        thread = VideoStreamThread(stream_url, stream_id)
        thread.frame_ready.connect(self.update_stream)
        self.stream_threads[stream_id] = thread
        thread.start()
        return True

    def update_stream(self, frame, stream_id):
        """Update the frame for a specific stream"""
        if stream_id in self.stream_windows:
            self.stream_windows[stream_id].update_frame(frame)

    def closeEvent(self, event):
        """Clean up threads when closing"""
        for thread in self.stream_threads.values():
            thread.stop()
            thread.wait()
        event.accept()

