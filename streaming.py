# streaming_manager.py
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
    """Enhanced streaming manager with support for multiple servers"""
    
    def __init__(self, security_system):
        self.security_system = security_system
        self.config = StreamConfig()
        self.active = False
        self.last_frame = None
        self.stream_lock = threading.Lock()
        self.error_count = {url: 0 for url in self.config.server_urls}
        self.max_errors = 10
        self.frame_count = 0
        self.last_successful_send = {url: None for url in self.config.server_urls}
        self.callbacks = {'on_error': [], 'on_success': [], 'on_stop': []}
        self.streaming_thread = None

    def add_callback(self, event: str, callback: Callable):
        if event in self.callbacks:
            self.callbacks[event].append(callback)

    def _trigger_callbacks(self, event: str, *args, **kwargs):
        for callback in self.callbacks.get(event, []):
            try:
                callback(*args, **kwargs)
            except Exception as e:
                logger.error(f"Callback error for {event}: {e}")

    def set_config(self, **kwargs):
        for key, value in kwargs.items():
            if key == 'server_urls':
                new_urls = value
                self.error_count = {url: self.error_count.get(url, 0) for url in new_urls}
                self.last_successful_send = {url: self.last_successful_send.get(url, None) for url in new_urls}
                setattr(self.config, key, value)
            elif hasattr(self.config, key):
                setattr(self.config, key, value)
        logger.info(f"Streaming config updated: {kwargs}")

    def get_streaming_stats(self) -> dict:
        return {
            'active': self.active,
            'frame_count': self.frame_count,
            'error_count': self.error_count,
            'last_successful_send': {
                url: self.last_successful_send[url].isoformat() if self.last_successful_send[url] else None
                for url in self.config.server_urls
            },
            'uptime': time.time() - getattr(self, 'start_time', time.time()),
            'servers': self.config.server_urls
        }

    def generate_frame_with_detection(self) -> Optional[bytes]:
        """Generate frame with face detection overlay"""
        try:
            frame = self.security_system.get_frame()
            if frame is None:
                return None
            
            self.last_frame = frame.copy()
            
            if self.config.detection_enabled:
                # Fix: Use process_face_recognition_enhanced
                results = self.security_system.process_face_recognition_enhanced(frame)
                self._draw_face_rectangles(frame, results)
                self._add_frame_info(frame)
            
            return self._frame_to_jpeg(frame)
            
        except Exception as e:
            logger.error(f"Frame generation error: {e}")
            self._trigger_callbacks('on_error', 'frame_generation', e)
            return None

    def _add_frame_info(self, frame):
        info_text = f"Frame: {self.frame_count} | Errors: {sum(self.error_count.values())}"
        cv2.putText(frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def _draw_face_rectangles(self, frame, results: List[FaceRecognitionResult]):
        for result in results:
            top, right, bottom, left = result.location
            color_map = {
                AccessResult.GRANTED: (0, 255, 0),
                AccessResult.DENIED: (0, 165, 255),
                AccessResult.UNKNOWN: (0, 0, 255)
            }
            color = color_map.get(result.access_result, (0, 0, 255))
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            label = f"{result.name} ({result.confidence:.1f}%)"
            cv2.putText(frame, label, (left + 6, bottom - 6), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

    def _frame_to_jpeg(self, frame) -> Optional[bytes]:
        if frame is None:
            return None
        try:
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
        with self.stream_lock:
            if self.active:
                logger.warning("Streaming already active")
                return False
            self.active = True
            self.error_count = {url: 0 for url in self.config.server_urls}
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
        logger.info("Streaming loop started")
        consecutive_failures = {url: 0 for url in self.config.server_urls}
        max_consecutive_failures = 5
        while self.active and all(self.error_count[url] < self.max_errors for url in self.config.server_urls):
            try:
                frame_bytes = self.generate_frame_with_detection()
                if frame_bytes:
                    results = self._send_frame_to_server(frame_bytes)
                    for server_url, success in results.items():
                        if success:
                            consecutive_failures[server_url] = 0
                            self.frame_count += 1
                        else:
                            consecutive_failures[server_url] += 1
                            if consecutive_failures[server_url] >= max_consecutive_failures:
                                backoff_time = min(2 ** consecutive_failures[server_url], 30)
                                logger.warning(f"Multiple consecutive failures for {server_url}, backing off for {backoff_time}s")
                                time.sleep(backoff_time)
                sleep_time = max(0.01, 1.0 / self.config.fps)
                time.sleep(sleep_time)
            except Exception as e:
                for url in self.config.server_urls:
                    self.error_count[url] += 1
                    consecutive_failures[url] += 1
                logger.error(f"Streaming error: {e}")
                self._trigger_callbacks('on_error', 'streaming_loop', e)
                time.sleep(1)
        if any(self.error_count[url] >= self.max_errors for url in self.config.server_urls):
            logger.error("Too many streaming errors for one or more servers, stopping stream")
            self.active = False
        logger.info("Streaming loop ended")

    def _send_frame_to_server(self, frame_bytes: bytes) -> dict:
        results = {}
        try:
            frame_b64 = base64.b64encode(frame_bytes).decode('utf-8')
            data = {
                "device_id": self.security_system.device_id,
                "timestamp": datetime.datetime.now().isoformat(),
                "frame": frame_b64,
                "detection_enabled": self.config.detection_enabled,
                "frame_count": self.frame_count,
                "quality": self.config.quality,
                "fps": self.config.fps
            }
            for server_url in self.config.server_urls:
                try:
                    response = requests.post(
                        f"{server_url}/receive_stream",
                        json=data,
                        timeout=self.config.timeout,
                        headers={'Content-Type': 'application/json'}
                    )
                    if response.status_code == 200:
                        self.error_count[server_url] = max(0, self.error_count[server_url] - 1)
                        self.last_successful_send[server_url] = datetime.datetime.now()
                        self._trigger_callbacks('on_success', response, server_url=server_url)
                        results[server_url] = True
                    else:
                        logger.warning(f"Server {server_url} responded with status: {response.status_code}")
                        self.error_count[server_url] += 1
                        results[server_url] = False
                except requests.Timeout:
                    logger.error(f"Request timeout when sending frame to {server_url}")
                    self.error_count[server_url] += 1
                    results[server_url] = False
                except requests.ConnectionError:
                    logger.error(f"Connection error when sending frame to {server_url}")
                    self.error_count[server_url] += 1
                    results[server_url] = False
                except requests.RequestException as e:
                    logger.error(f"Failed to send frame to {server_url}: {e}")
                    self.error_count[server_url] += 1
                    results[server_url] = False
                except Exception as e:
                    logger.error(f"Unexpected error sending frame to {server_url}: {e}")
                    self.error_count[server_url] += 1
                    results[server_url] = False
            return results
        except Exception as e:
            logger.error(f"General error preparing frame for servers: {e}")
            return {url: False for url in self.config.server_urls}

    def is_healthy(self) -> bool:
        if not self.active:
            return False
        for server_url in self.config.server_urls:
            if self.last_successful_send[server_url]:
                time_since_success = datetime.datetime.now() - self.last_successful_send[server_url]
                if time_since_success.total_seconds() <= 60:
                    if self.error_count[server_url] <= self.max_errors * 0.8:
                        return True
        return False

    def restart_if_unhealthy(self) -> bool:
        if not self.is_healthy():
            logger.info("Streaming appears unhealthy, attempting restart")
            self.stop_streaming()
            time.sleep(2)
            return self.start_streaming()
        return False