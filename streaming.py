import threading
import logging
import time
import datetime
import cv2
import requests
import base64
from typing import Optional, List, Callable
from config import StreamConfig
from models import FaceRecognitionResult
from enums import AccessResult

logger = logging.getLogger(__name__)

class StreamingManager:
    """Enhanced streaming manager with better error handling and additional features"""
    
    def __init__(self, security_system):
        self.security_system = security_system
        self.config = StreamConfig()
        self.active = False
        self.last_frame = None
        self.streaming_thread = None
        self.stream_lock = threading.Lock()
        self.error_count = 0
        self.max_errors = 10
        self.frame_count = 0
        self.last_successful_send = None
        self.callbacks = {'on_error': [], 'on_success': [], 'on_stop': []}

    def add_callback(self, event: str, callback: Callable):
        """Add callback for events (on_error, on_success, on_stop)"""
        if event in self.callbacks:
            self.callbacks[event].append(callback)

    def _trigger_callbacks(self, event: str, *args, **kwargs):
        """Trigger callbacks for specific events"""
        for callback in self.callbacks.get(event, []):
            try:
                callback(*args, **kwargs)
            except Exception as e:
                logger.error(f"Callback error for {event}: {e}")

    def set_config(self, **kwargs):
        """Update streaming configuration"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        logger.info(f"Streaming config updated: {kwargs}")

    def get_streaming_stats(self) -> dict:
        """Get current streaming statistics"""
        return {
            'active': self.active,
            'frame_count': self.frame_count,
            'error_count': self.error_count,
            'last_successful_send': self.last_successful_send,
            'uptime': time.time() - getattr(self, 'start_time', time.time())
        }

    def generate_frame_with_detection(self) -> Optional[bytes]:
        """Generate frame with face detection overlay"""
        try:
            frame = self.security_system.get_frame()
            if frame is None:
                return None
            
            # Store frame for debugging/monitoring
            self.last_frame = frame.copy()
            
            if self.config.detection_enabled:
                results = self.security_system.process_face_recognition(frame)
                self._draw_face_rectangles(frame, results)
                
                # Add frame info overlay
                self._add_frame_info(frame)
            
            return self._frame_to_jpeg(frame)
            
        except Exception as e:
            logger.error(f"Frame generation error: {e}")
            self._trigger_callbacks('on_error', 'frame_generation', e)
            return None

    def _add_frame_info(self, frame):
        """Add frame information overlay"""
        info_text = f"Frame: {self.frame_count} | Errors: {self.error_count}"
        cv2.putText(frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

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
        """Convert frame to JPEG bytes with error handling"""
        if frame is None:
            return None
        
        try:
            # Validate frame dimensions
            if frame.shape[0] == 0 or frame.shape[1] == 0:
                logger.warning("Invalid frame dimensions")
                return None
                
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.config.quality]
            success, buffer = cv2.imencode('.jpg', frame, encode_param)
            
            if not success:
                logger.error("Failed to encode frame to JPEG")
                return None
                
            return buffer.tobytes()
            
        except Exception as e:
            logger.error(f"Frame encoding error: {e}")
            return None

    def start_streaming(self) -> bool:
        """Start streaming to server"""
        with self.stream_lock:
            if self.active:
                logger.warning("Streaming already active")
                return False
            
            # Reset counters
            self.active = True
            self.error_count = 0
            self.frame_count = 0
            self.start_time = time.time()
            
            self.streaming_thread = threading.Thread(
                target=self._streaming_loop,
                daemon=True,
                name="Streaming-Thread"
            )
            self.streaming_thread.start()
            
            logger.info("Streaming started")
            return True

    def stop_streaming(self, timeout: float = 5.0) -> bool:
        """Stop streaming with configurable timeout"""
        with self.stream_lock:
            if not self.active:
                logger.warning("Streaming not active")
                return False
            
            self.active = False
            
            if self.streaming_thread and self.streaming_thread.is_alive():
                self.streaming_thread.join(timeout=timeout)
                
                if self.streaming_thread.is_alive():
                    logger.warning("Streaming thread did not stop within timeout")
            
            self._trigger_callbacks('on_stop')
            logger.info("Streaming stopped")
            return True

    def _streaming_loop(self):
        """Main streaming loop with enhanced error handling"""
        logger.info("Streaming loop started")
        consecutive_failures = 0
        max_consecutive_failures = 5
        
        while self.active and self.error_count < self.max_errors:
            try:
                frame_bytes = self.generate_frame_with_detection()
                
                if frame_bytes and self.config.server_url:
                    success = self._send_frame_to_server(frame_bytes)
                    
                    if success:
                        consecutive_failures = 0
                        self.frame_count += 1
                    else:
                        consecutive_failures += 1
                        
                        # Implement exponential backoff for consecutive failures
                        if consecutive_failures >= max_consecutive_failures:
                            backoff_time = min(2 ** consecutive_failures, 30)
                            logger.warning(f"Multiple consecutive failures, backing off for {backoff_time}s")
                            time.sleep(backoff_time)
                
                # Dynamic sleep based on FPS
                sleep_time = max(0.01, 1.0 / self.config.fps)
                time.sleep(sleep_time)
                
            except Exception as e:
                self.error_count += 1
                consecutive_failures += 1
                logger.error(f"Streaming error ({self.error_count}/{self.max_errors}): {e}")
                self._trigger_callbacks('on_error', 'streaming_loop', e)
                time.sleep(1)
        
        if self.error_count >= self.max_errors:
            logger.error("Too many streaming errors, stopping stream")
            self.active = False
        
        logger.info("Streaming loop ended")

    def _send_frame_to_server(self, frame_bytes: bytes) -> bool:
        """Send frame to streaming server with improved error handling"""
        try:
            frame_b64 = base64.b64encode(frame_bytes).decode('utf-8')
            
            # Add additional metadata
            data = {
                "device_id": self.security_system.device_id,
                "timestamp": datetime.datetime.now().isoformat(),
                "frame": frame_b64,
                "detection_enabled": self.config.detection_enabled,
                "frame_count": self.frame_count,
                "quality": self.config.quality,
                "fps": self.config.fps
            }
            
            response = requests.post(
                f"{self.config.server_url}/receive_stream",
                json=data,
                timeout=self.config.get('timeout', 5),
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                # Reduce error count on success (recovery mechanism)
                self.error_count = max(0, self.error_count - 1)
                self.last_successful_send = datetime.datetime.now()
                self._trigger_callbacks('on_success', response)
                return True
            else:
                logger.warning(f"Server responded with status: {response.status_code}")
                return False
                
        except requests.Timeout:
            logger.error("Request timeout when sending frame to server")
            return False
        except requests.ConnectionError:
            logger.error("Connection error when sending frame to server")
            return False
        except requests.RequestException as e:
            logger.error(f"Failed to send frame to server: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending frame: {e}")
            return False

    def is_healthy(self) -> bool:
        """Check if streaming is healthy"""
        if not self.active:
            return False
            
        # Check if we've had recent successful sends
        if self.last_successful_send:
            time_since_success = datetime.datetime.now() - self.last_successful_send
            if time_since_success.total_seconds() > 60:  # 1 minute threshold
                return False
                
        # Check error rate
        if self.error_count > self.max_errors * 0.8:  # 80% of max errors
            return False
            
        return True

    def restart_if_unhealthy(self) -> bool:
        """Restart streaming if it's unhealthy"""
        if not self.is_healthy():
            logger.info("Streaming appears unhealthy, attempting restart")
            self.stop_streaming()
            time.sleep(2)  # Brief pause before restart
            return self.start_streaming()
        return False