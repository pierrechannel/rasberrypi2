import threading
import logging
import time
import datetime
import cv2
import requests
import base64
from typing import Optional, List
from config import StreamConfig
from models import FaceRecognitionResult
from enums import AccessResult

logger = logging.getLogger(__name__)

class StreamingManager:
    """Enhanced streaming manager with better error handling"""
    
    def __init__(self, security_system):
        self.security_system = security_system
        self.config = StreamConfig()
        self.active = False
        self.last_frame = None
        self.streaming_thread = None
        self.stream_lock = threading.Lock()
        self.error_count = 0
        self.max_errors = 10

    def set_config(self, **kwargs):
        """Update streaming configuration"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        logger.info(f"Streaming config updated: {kwargs}")

    def generate_frame_with_detection(self) -> Optional[bytes]:
        """Generate frame with face detection overlay"""
        try:
            frame = self.security_system.get_frame()
            if frame is None:
                return None
            
            if self.config.detection_enabled:
                results = self.security_system.process_face_recognition(frame)
                self._draw_face_rectangles(frame, results)
            
            return self._frame_to_jpeg(frame)
            
        except Exception as e:
            logger.error(f"Frame generation error: {e}")
            return None

    def _draw_face_rectangles(self, frame, results: List[FaceRecognitionResult]):
        """Draw face detection rectangles on frame"""
        for result in results:
            top, right, bottom, left = result.location
            
            # Color based on access result
            color_map = {
                AccessResult.GRANTED: (0, 255, 0),  # Green
                AccessResult.DENIED: (0, 165, 255),  # Orange
                AccessResult.UNKNOWN: (0, 0, 255)   # Red
            }
            color = color_map.get(result.access_result, (0, 0, 255))
            
            # Draw rectangle and label
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            
            label = f"{result.name} ({result.confidence:.1f}%)"
            cv2.putText(frame, label, (left + 6, bottom - 6), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

    def _frame_to_jpeg(self, frame) -> Optional[bytes]:
        """Convert frame to JPEG bytes"""
        if frame is None:
            return None
        
        try:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.config.quality]
            _, buffer = cv2.imencode('.jpg', frame, encode_param)
            return buffer.tobytes()
        except Exception as e:
            logger.error(f"Frame encoding error: {e}")
            return None

    def start_streaming(self) -> bool:
        """Start streaming to server"""
        with self.stream_lock:
            if self.active:
                return False
            
            self.active = True
            self.error_count = 0
            
            self.streaming_thread = threading.Thread(
                target=self._streaming_loop,
                daemon=True,
                name="Streaming-Thread"
            )
            self.streaming_thread.start()
            
            logger.info("Streaming started")
            return True

    def stop_streaming(self) -> bool:
        """Stop streaming"""
        with self.stream_lock:
            if not self.active:
                return False
            
            self.active = False
            
            if self.streaming_thread and self.streaming_thread.is_alive():
                self.streaming_thread.join(timeout=5)
            
            logger.info("Streaming stopped")
            return True

    def _streaming_loop(self):
        """Main streaming loop"""
        logger.info("Streaming loop started")
        
        while self.active and self.error_count < self.max_errors:
            try:
                frame_bytes = self.generate_frame_with_detection()
                if frame_bytes and self.config.server_url:
                    self._send_frame_to_server(frame_bytes)
                
                time.sleep(1.0 / self.config.fps)
                
            except Exception as e:
                self.error_count += 1
                logger.error(f"Streaming error ({self.error_count}/{self.max_errors}): {e}")
                time.sleep(1)
        
        if self.error_count >= self.max_errors:
            logger.error("Too many streaming errors, stopping stream")
            self.active = False
        
        logger.info("Streaming loop ended")

    def _send_frame_to_server(self, frame_bytes: bytes):
        """Send frame to streaming server"""
        try:
            frame_b64 = base64.b64encode(frame_bytes).decode('utf-8')
            data = {
                "device_id": self.security_system.device_id,
                "timestamp": datetime.datetime.now().isoformat(),
                "frame": frame_b64,
                "detection_enabled": self.config.detection_enabled
            }
            
            response = requests.post(
                f"{self.config.server_url}/receive_stream",
                json=data,
                timeout=5
            )
            
            if response.status_code == 200:
                self.error_count = max(0, self.error_count - 1)  # Reduce error count on success
            else:
                logger.warning(f"Server responded with status: {response.status_code}")
                
        except requests.RequestException as e:
            logger.error(f"Failed to send frame to server: {e}")
            raise