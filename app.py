import time
import threading
import logging
from threading import Timer, Event
from collections import deque
from flask import Flask, request, jsonify, Response
from contextlib import contextmanager
import datetime
import cv2
import json
import requests
import base64
from io import BytesIO
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any
from enum import Enum
import signal

# Optional imports with graceful fallback
try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    print("WARNING: RPi.GPIO not available. Door lock control will be simulated.")

try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("WARNING: pyttsx3 not available. TTS will be simulated.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AccessResult(Enum):
    """Enumeration for access control results"""
    GRANTED = "granted"
    DENIED = "denied"
    UNKNOWN = "unknown"

@dataclass
class FaceRecognitionResult:
    """Data class for face recognition results"""
    person_id: str
    name: str
    confidence: float
    location: Tuple[int, int, int, int]  # (top, right, bottom, left)
    access_result: AccessResult

@dataclass
class StreamConfig:
    """Configuration for video streaming"""
    fps: int = 10
    quality: int = 80
    detection_enabled: bool = True
    server_url: Optional[str] = None

class TTSManager:
    """Enhanced Text-to-Speech Manager with better error handling and performance"""
    
    VOICE_MESSAGES = {
        "welcome": "Welcome, {name}! Access granted.",
        "access_denied": "Access denied. Please contact administrator.",
        "unknown_person": "Unknown person detected. Please identify yourself.",
        "door_unlocked": "Door unlocked for {duration} seconds.",
        "door_locked": "Door locked. System secured.",
        "system_startup": "Security system activated.",
        "system_shutdown": "Security system shutting down.",
        "camera_error": "Camera error detected. Please check connection.",
        "streaming_started": "Video streaming started.",
        "streaming_stopped": "Video streaming stopped.",
        "recognition_cooldown": "Recognition system in cooldown mode.",
        "backend_offline": "Backend connection lost. Operating in offline mode.",
        "sync_successful": "Synchronization with backend completed successfully."
    }
    
    def __init__(self):
        self.is_active = False
        self.tts_queue = deque()
        self.tts_lock = threading.Lock()
        self.shutdown_event = Event()
        self.playback_thread = None
        self.tts_engine = None
        
        if TTS_AVAILABLE:
            self._setup_tts_engine()
        else:
            logger.warning("TTS in simulation mode")

    def _setup_tts_engine(self) -> bool:
        """Setup TTS engine with error handling"""
        try:
            self.tts_engine = pyttsx3.init(driverName='espeak')
            
            # Configure TTS properties
            self.tts_engine.setProperty('rate', 150)
            self.tts_engine.setProperty('volume', 0.9)
            
            # Set English voice if available
            voices = self.tts_engine.getProperty('voices')
            if voices:
                for voice in voices:
                    if any(lang in voice.name.lower() for lang in ['english', 'en']):
                        self.tts_engine.setProperty('voice', voice.id)
                        break
            
            self.is_active = True
            logger.info("TTS system initialized successfully")
            
            # Start playback thread
            self.playback_thread = threading.Thread(
                target=self._tts_playback_loop, 
                daemon=True,
                name="TTS-Playback"
            )
            self.playback_thread.start()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize TTS engine: {e}")
            self.is_active = False
            return False

    def speak(self, message_key: str, priority: bool = False, **kwargs) -> bool:
        """Speak a predefined message with optional parameters"""
        if not self.is_active:
            logger.debug(f"TTS simulation: {message_key}")
            return False
            
        if message_key not in self.VOICE_MESSAGES:
            logger.warning(f"Unknown message key: {message_key}")
            return False
        
        try:
            message = self.VOICE_MESSAGES[message_key]
            if kwargs:
                message = message.format(**kwargs)
            
            return self._queue_message(message, priority)
            
        except KeyError as e:
            logger.warning(f"Missing parameter for message {message_key}: {e}")
            return False

    def speak_custom(self, message: str, priority: bool = False) -> bool:
        """Speak a custom message"""
        if not self.is_active:
            logger.debug(f"TTS simulation: {message}")
            return False
        
        return self._queue_message(message, priority)

    def _queue_message(self, message: str, priority: bool = False) -> bool:
        """Queue a message for TTS playback"""
        try:
            with self.tts_lock:
                if priority:
                    self.tts_queue.appendleft(message)
                else:
                    self.tts_queue.append(message)
            
            logger.debug(f"TTS message queued: {message}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to queue TTS message: {e}")
            return False

    def _tts_playback_loop(self):
        """Main TTS playback loop"""
        logger.info("TTS playback loop started")
        
        while not self.shutdown_event.is_set():
            try:
                message = None
                with self.tts_lock:
                    if self.tts_queue:
                        message = self.tts_queue.popleft()
                
                if message:
                    try:
                        self.tts_engine.say(message)
                        self.tts_engine.runAndWait()
                    except Exception as e:
                        logger.error(f"TTS playback error: {e}")
                
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"TTS loop error: {e}")
                time.sleep(1)
        
        logger.info("TTS playback loop stopped")

    def clear_queue(self):
        """Clear all pending TTS messages"""
        with self.tts_lock:
            self.tts_queue.clear()
        logger.debug("TTS queue cleared")

    def cleanup(self):
        """Cleanup TTS resources"""
        logger.info("Shutting down TTS system")
        self.shutdown_event.set()
        self.clear_queue()
        
        if self.playback_thread and self.playback_thread.is_alive():
            self.playback_thread.join(timeout=2)
        
        if self.tts_engine:
            try:
                self.tts_engine.stop()
            except:
                pass
        
        self.is_active = False

class DoorLockController:
    """Enhanced door lock controller with TTS integration"""
    
    def __init__(self, relay_pin: int = 18, led_green_pin: int = 16, 
                 led_red_pin: int = 20, buzzer_pin: int = 21, 
                 tts_manager: Optional[TTSManager] = None):
        self.relay_pin = relay_pin
        self.led_green_pin = led_green_pin
        self.led_red_pin = led_red_pin
        self.buzzer_pin = buzzer_pin
        self.lock_timer = None
        self.is_door_open = False
        self.tts_manager = tts_manager
        self.lock = threading.Lock()
        
        if GPIO_AVAILABLE:
            self._setup_gpio()
        else:
            logger.info("GPIO simulation mode activated")

    def _setup_gpio(self):
        """Initialize GPIO pins"""
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            
            # Setup output pins
            for pin in [self.relay_pin, self.led_green_pin, self.led_red_pin, self.buzzer_pin]:
                GPIO.setup(pin, GPIO.OUT)
                GPIO.output(pin, GPIO.LOW)
            
            # Start in locked state
            self._set_locked_state()
            logger.info("GPIO pins initialized successfully")
            
        except Exception as e:
            logger.error(f"GPIO setup failed: {e}")

    def unlock_door(self, duration: int = 5, name: Optional[str] = None) -> bool:
        """Unlock door with TTS announcement"""
        with self.lock:
            try:
                if GPIO_AVAILABLE:
                    GPIO.output(self.relay_pin, GPIO.HIGH)
                    GPIO.output(self.led_green_pin, GPIO.HIGH)
                    GPIO.output(self.led_red_pin, GPIO.LOW)
                
                self.is_door_open = True
                logger.info(f"Door unlocked for {duration} seconds")
                
                # TTS announcements
                if self.tts_manager:
                    if name and name != "Unknown":
                        self.tts_manager.speak("welcome", name=name)
                    self.tts_manager.speak("door_unlocked", duration=duration)
                
                # Cancel existing timer
                if self.lock_timer:
                    self.lock_timer.cancel()
                
                # Set new lock timer
                self.lock_timer = Timer(duration, self.lock_door)
                self.lock_timer.start()
                
                return True
                
            except Exception as e:
                logger.error(f"Failed to unlock door: {e}")
                return False

    def lock_door(self) -> bool:
        """Lock door with TTS announcement"""
        with self.lock:
            try:
                self._set_locked_state()
                self.is_door_open = False
                logger.info("Door locked")
                
                if self.tts_manager:
                    self.tts_manager.speak("door_locked")
                
                return True
                
            except Exception as e:
                logger.error(f"Failed to lock door: {e}")
                return False

    def _set_locked_state(self):
        """Set GPIO pins to locked state"""
        if GPIO_AVAILABLE:
            GPIO.output(self.relay_pin, GPIO.LOW)
            GPIO.output(self.led_green_pin, GPIO.LOW)
            GPIO.output(self.led_red_pin, GPIO.HIGH)

    def handle_access_denied(self, reason: str = "access_denied"):
        """Handle access denied with TTS instead of beep"""
        logger.warning(f"Access denied: {reason}")
        if self.tts_manager:
            self.tts_manager.speak(reason, priority=True)

    def handle_unknown_person(self):
        """Handle unknown person detection with TTS"""
        logger.info("Unknown person detected")
        if self.tts_manager:
            self.tts_manager.speak("unknown_person", priority=True)

    def emergency_unlock(self) -> bool:
        """Emergency unlock without timer"""
        with self.lock:
            if self.lock_timer:
                self.lock_timer.cancel()
            
            if GPIO_AVAILABLE:
                GPIO.output(self.relay_pin, GPIO.HIGH)
                GPIO.output(self.led_green_pin, GPIO.HIGH)
                GPIO.output(self.led_red_pin, GPIO.LOW)
            
            self.is_door_open = True
            logger.warning("Emergency door unlock activated")
            
            if self.tts_manager:
                self.tts_manager.speak_custom("Emergency unlock activated", priority=True)
            
            return True

    def cleanup(self):
        """Cleanup door lock resources"""
        logger.info("Cleaning up door lock controller")
        
        if self.lock_timer:
            self.lock_timer.cancel()
        
        if GPIO_AVAILABLE:
            self.lock_door()
            GPIO.cleanup()

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

class SecuritySystem:
    """Main security system class with enhanced error handling"""
    
    def __init__(self, device_id: str = "RPI_001"):
        self.device_id = device_id
        self.logger = logging.getLogger(f"{__name__}.{device_id}")
        
        # Initialize components
        self.tts_manager = TTSManager()
        self.door_lock = DoorLockController(tts_manager=self.tts_manager)
        self.streaming_manager = StreamingManager(self)
        
        # Camera setup
        self.video_capture = None
        self._setup_camera()
        
        # Recognition settings
        self.known_face_names = ["John", "Jane", "Admin"]
        self.last_recognition_time = 0
        self.recognition_cooldown = 5
        
        # Backend connection
        self.backend_api_url = "https://example.com/api"
        self.backend_online = True
        self.offline_logs = []
        
        # Startup announcement
        if self.tts_manager:
            self.tts_manager.speak("system_startup")
        
        self.logger.info("Security system initialized")

    def _setup_camera(self):
        """Initialize camera with error handling"""
        try:
            self.video_capture = cv2.VideoCapture(0)
            if not self.video_capture.isOpened():
                raise Exception("Camera not accessible")
            
            # Set camera properties
            self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.video_capture.set(cv2.CAP_PROP_FPS, 30)
            
            self.logger.info("Camera initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Camera setup failed: {e}")
            if self.tts_manager:
                self.tts_manager.speak("camera_error")

    def get_frame(self) -> Optional[any]:
        """Get current camera frame"""
        if not self.video_capture or not self.video_capture.isOpened():
            return None
        
        try:
            ret, frame = self.video_capture.read()
            return frame if ret else None
        except Exception as e:
            self.logger.error(f"Frame capture error: {e}")
            return None

    def process_face_recognition(self, frame) -> List[FaceRecognitionResult]:
        """Process face recognition on frame"""
        # Mock implementation - replace with actual face recognition
        results = []
        
        # Simulate face detection
        if frame is not None:
            # Mock face location
            height, width = frame.shape[:2]
            face_location = (100, 300, 300, 100)  # (top, right, bottom, left)
            
            # Mock recognition result
            import random
            names = self.known_face_names + ["Unknown"]
            name = random.choice(names)
            confidence = random.uniform(75, 95) if name != "Unknown" else random.uniform(30, 60)
            
            if name in self.known_face_names:
                access_result = AccessResult.GRANTED
            else:
                access_result = AccessResult.UNKNOWN
            
            result = FaceRecognitionResult(
                person_id=f"ID_{hash(name) % 1000}",
                name=name,
                confidence=confidence,
                location=face_location,
                access_result=access_result
            )
            results.append(result)
        
        return results

    @contextmanager
    def recognition_cooldown_check(self):
        """Context manager for recognition cooldown"""
        current_time = time.time()
        if (current_time - self.last_recognition_time) < self.recognition_cooldown:
            remaining = self.recognition_cooldown - (current_time - self.last_recognition_time)
            if self.tts_manager:
                self.tts_manager.speak("recognition_cooldown")
            raise Exception(f"Recognition cooldown active. {remaining:.1f}s remaining")
        
        yield
        
        self.last_recognition_time = current_time

    def process_access_attempt(self, frame) -> Dict[str, Any]:
        """Process access attempt with enhanced logic"""
        try:
            with self.recognition_cooldown_check():
                results = self.process_face_recognition(frame)
                response_data = {
                    "faces_detected": len(results),
                    "results": [],
                    "timestamp": datetime.datetime.now().isoformat(),
                    "device_id": self.device_id
                }
                
                for result in results:
                    # Handle access based on result
                    if result.access_result == AccessResult.GRANTED:
                        self.door_lock.unlock_door(duration=5, name=result.name)
                    elif result.access_result == AccessResult.UNKNOWN:
                        self.door_lock.handle_unknown_person()
                    else:
                        self.door_lock.handle_access_denied()
                    
                    # Log attempt
                    self._log_access_attempt(result, frame)
                    
                    # Add to response
                    response_data["results"].append({
                        "face_index": len(response_data["results"]),
                        "person_id": result.person_id,
                        "name": result.name,
                        "access_granted": result.access_result == AccessResult.GRANTED,
                        "confidence": result.confidence
                    })
                
                return response_data
                
        except Exception as e:
            self.logger.error(f"Access attempt processing error: {e}")
            raise

    def _log_access_attempt(self, result: FaceRecognitionResult, frame):
        """Log access attempt"""
        try:
            image_base64 = self._image_to_base64(frame)
            log_entry = {
                "person_id": result.person_id,
                "name": result.name,
                "access_granted": result.access_result == AccessResult.GRANTED,
                "confidence": result.confidence,
                "timestamp": datetime.datetime.now().isoformat(),
                "image": image_base64
            }
            self.offline_logs.append(log_entry)
            self.logger.info(f"Access attempt logged: {result.name}")
        except Exception as e:
            self.logger.error(f"Failed to log access attempt: {e}")

    def _image_to_base64(self, frame) -> str:
        """Convert frame to base64 string"""
        try:
            _, buffer = cv2.imencode('.jpg', frame)
            return base64.b64encode(buffer).decode('utf-8')
        except Exception as e:
            self.logger.error(f"Image encoding error: {e}")
            return ""

    def sync_with_backend(self) -> bool:
        """Sync offline logs with backend"""
        try:
            # Mock backend sync
            if self.offline_logs:
                self.logger.info(f"Syncing {len(self.offline_logs)} logs with backend")
                # Clear logs after successful sync
                self.offline_logs.clear()
                if self.tts_manager:
                    self.tts_manager.speak("sync_successful")
            return True
        except Exception as e:
            self.logger.error(f"Backend sync failed: {e}")
            if self.tts_manager:
                self.tts_manager.speak("backend_offline")
            return False

    def add_person(self, name: str, image: Optional[any] = None) -> Dict[str, Any]:
        """Add a new person to the known face names list."""
        try:
            if not name or not isinstance(name, str):
                raise ValueError("Valid name is required")
            
            if name in self.known_face_names:
                raise ValueError(f"Person '{name}' already exists in the system")
            
            # Add name to known_face_names
            self.known_face_names.append(name)
            
            # Log the addition
            log_entry = {
                "person_id": f"ID_{hash(name) % 1000}",
                "name": name,
                "timestamp": datetime.datetime.now().isoformat(),
                "action": "person_added"
            }
            
            # If an image is provided, store it (for future face recognition integration)
            if image is not None:
                log_entry["image"] = self._image_to_base64(image)
            
            self.offline_logs.append(log_entry)
            self.logger.info(f"Added new person: {name}")
            
            # Announce via TTS
            if self.tts_manager:
                self.tts_manager.speak_custom(f"New person {name} added to the system", priority=True)
            
            # Attempt to sync with backend
            self.sync_with_backend()
            
            return {
                "success": True,
                "message": f"Person '{name}' added successfully",
                "person_id": log_entry["person_id"],
                "known_persons_count": len(self.known_face_names)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to add person: {e}")
            return {
                "success": False,
                "message": f"Failed to add person: {str(e)}"
            }

    def cleanup(self):
        """Cleanup all system resources"""
        self.logger.info("Shutting down security system")
        
        if self.tts_manager:
            self.tts_manager.speak("system_shutdown")
            time.sleep(2)  # Allow TTS to complete
        
        # Stop streaming
        self.streaming_manager.stop_streaming()
        
        # Cleanup camera
        if self.video_capture:
            self.video_capture.release()
        
        # Cleanup door lock
        self.door_lock.cleanup()
        
        # Cleanup TTS
        if self.tts_manager:
            self.tts_manager.cleanup()
        
        self.logger.info("Security system shutdown complete")

# Flask Application Setup
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Initialize security system
security_system = SecuritySystem()

# Route definitions
@app.route('/health', methods=['GET'])
def health_check():
    """System health check endpoint"""
    return jsonify({
        "status": "healthy",
        "device_id": security_system.device_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "known_persons": len(security_system.known_face_names),
        "camera_status": "active" if security_system.video_capture 
                        and security_system.video_capture.isOpened() else "inactive",
        "tts_active": security_system.tts_manager.is_active,
        "backend_online": security_system.backend_online,
        "offline_logs_count": len(security_system.offline_logs),
        "streaming_active": security_system.streaming_manager.active,
        "door_status": "open" if security_system.door_lock.is_door_open else "locked"
    })

@app.route('/capture', methods=['GET'])
def capture_and_recognize():
    """Capture image and perform face recognition"""
    try:
        frame = security_system.get_frame()
        if frame is None:
            return jsonify({"error": "Unable to capture image"}), 500
        
        result = security_system.process_access_attempt(frame)
        return jsonify(result)
        
    except Exception as e:
        if "cooldown" in str(e).lower():
            return jsonify({"error": str(e)}), 429
        
        security_system.logger.error(f"Capture error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/stream/config', methods=['GET', 'POST'])
def stream_config():
    """Get or update streaming configuration"""
    if request.method == 'GET':
        return jsonify({
            "fps": security_system.streaming_manager.config.fps,
            "quality": security_system.streaming_manager.config.quality,
            "detection_enabled": security_system.streaming_manager.config.detection_enabled,
            "server_url": security_system.streaming_manager.config.server_url,
            "active": security_system.streaming_manager.active
        })
    
    elif request.method == 'POST':
        try:
            data = request.get_json() or {}
            
            # Update configuration
            config_updates = {}
            for key in ['fps', 'quality', 'detection_enabled', 'server_url']:
                if key in data:
                    if key == 'fps' and data[key]:
                        config_updates[key] = max(1, min(30, int(data[key])))
                    elif key == 'quality' and data[key]:
                        config_updates[key] = max(10, min(100, int(data[key])))
                    else:
                        config_updates[key] = data[key]
            
            if config_updates:
                security_system.streaming_manager.set_config(**config_updates)
            
            return jsonify({
                "message": "Configuration updated",
                "config": {
                    "fps": security_system.streaming_manager.config.fps,
                    "quality": security_system.streaming_manager.config.quality,
                    "detection_enabled": security_system.streaming_manager.config.detection_enabled,
                    "server_url": security_system.streaming_manager.config.server_url
                }
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

@app.route('/stream/live')
def live_stream():
    """Live video stream endpoint"""
    def generate():
        # Ensure streaming is active for local viewing
        if not security_system.streaming_manager.active:
            security_system.streaming_manager.active = True
        
        try:
            while True:
                frame_bytes = security_system.streaming_manager.generate_frame_with_detection()
                if frame_bytes:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                time.sleep(1.0 / security_system.streaming_manager.config.fps)
        except GeneratorExit:
            pass
        except Exception as e:
            security_system.logger.error(f"Live stream error: {e}")
    
    return Response(
        generate(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/stream/snapshot', methods=['GET'])
def get_snapshot():
    """Get current frame snapshot"""
    try:
        frame_bytes = security_system.streaming_manager.generate_frame_with_detection()
        if frame_bytes is None:
            return jsonify({"error": "Unable to capture image"}), 500
        
        frame_b64 = base64.b64encode(frame_bytes).decode('utf-8')
        return jsonify({
            "image": frame_b64,
            "timestamp": datetime.datetime.now().isoformat(),
            "device_id": security_system.device_id,
            "detection_enabled": security_system.streaming_manager.config.detection_enabled
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/status', methods=['GET'])
def get_system_status():
    """Get comprehensive system status"""
    return jsonify({
        "device_id": security_system.device_id,
        "system_time": datetime.datetime.now().isoformat(),
        "known_persons": len(security_system.known_face_names),
        "camera_active": security_system.video_capture and security_system.video_capture.isOpened(),
        "tts_active": security_system.tts_manager.is_active,
        "backend_online": security_system.backend_online,
        "offline_logs_pending": len(security_system.offline_logs),
        "last_recognition": security_system.last_recognition_time,
        "backend_url": security_system.backend_api_url,
        "recognition_cooldown": security_system.recognition_cooldown,
        "door": {
            "is_open": security_system.door_lock.is_door_open,
            "gpio_available": GPIO_AVAILABLE
        },
        "streaming": {
            "active": security_system.streaming_manager.active,
            "fps": security_system.streaming_manager.config.fps,
            "quality": security_system.streaming_manager.config.quality,
            "detection_enabled": security_system.streaming_manager.config.detection_enabled,
            "server_url": security_system.streaming_manager.config.server_url,
            "error_count": security_system.streaming_manager.error_count
        },
        "tts": {
            "active": security_system.tts_manager.is_active,
            "queue_size": len(security_system.tts_manager.tts_queue),
            "available_messages": list(security_system.tts_manager.VOICE_MESSAGES.keys())
        }
    })

@app.route('/sync', methods=['POST'])
def manual_sync():
    """Manually sync with backend"""
    try:
        success = security_system.sync_with_backend()
        return jsonify({
            "success": success,
            "message": "Synchronization completed" if success else "Synchronization failed",
            "known_persons": len(security_system.known_face_names),
            "offline_logs_processed": 0 if success else len(security_system.offline_logs)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/logs/offline', methods=['GET'])
def get_offline_logs():
    """Get offline access logs"""
    return jsonify({
        "offline_logs": security_system.offline_logs,
        "count": len(security_system.offline_logs),
        "device_id": security_system.device_id
    })

@app.route('/logs/clear', methods=['POST'])
def clear_offline_logs():
    """Clear offline logs"""
    try:
        count = len(security_system.offline_logs)
        security_system.offline_logs.clear()
        return jsonify({
            "message": f"Cleared {count} offline logs",
            "remaining_logs": len(security_system.offline_logs)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/tts/queue/clear', methods=['POST'])
def clear_tts_queue():
    """Clear TTS message queue"""
    try:
        security_system.tts_manager.clear_queue()
        return jsonify({
            "message": "TTS queue cleared",
            "success": True
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/tts/messages', methods=['GET'])
def get_tts_messages():
    """Get available TTS messages"""
    return jsonify({
        "predefined_messages": security_system.tts_manager.VOICE_MESSAGES,
        "queue_size": len(security_system.tts_manager.tts_queue),
        "tts_active": security_system.tts_manager.is_active
    })

@app.route('/door/lock', methods=['POST'])
def manual_lock():
    """Manually lock door"""
    try:
        success = security_system.door_lock.lock_door()
        return jsonify({
            "success": success,
            "message": "Door locked" if success else "Failed to lock door",
            "door_status": "locked" if success else "unknown"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/door/unlock', methods=['POST'])
def manual_unlock():
    """Manually unlock door"""
    try:
        data = request.get_json() or {}
        duration = data.get('duration', 5)
        name = data.get('name', 'Manual Override')
        
        success = security_system.door_lock.unlock_door(duration=duration, name=name)
        return jsonify({
            "success": success,
            "message": f"Door unlocked for {duration} seconds" if success else "Failed to unlock door",
            "duration": duration,
            "door_status": "unlocked" if success else "unknown"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/door/emergency_unlock', methods=['POST'])
def emergency_unlock():
    """Emergency door unlock endpoint"""
    try:
        success = security_system.door_lock.emergency_unlock()
        return jsonify({
            "success": success,
            "message": "Emergency unlock activated" if success else "Emergency unlock failed"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/announce', methods=['POST'])
def make_announcement():
    """Make TTS announcement"""
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({"error": "Message required"}), 400
        
        message = data['message']
        message_type = data.get('type', 'custom')
        priority = data.get('priority', False)
        
        if message_type == 'predefined':
            success = security_system.tts_manager.speak(message, priority=priority)
        else:
            success = security_system.tts_manager.speak_custom(message, priority=priority)
        
        return jsonify({
            "message": "Announcement queued" if success else "TTS not available",
            "success": success
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/stream/start', methods=['POST'])
def start_streaming():
    """Start video streaming"""
    try:
        data = request.get_json() or {}
        
        # Update streaming configuration
        config_updates = {}
        for key in ['fps', 'quality', 'detection_enabled', 'server_url']:
            if key in data:
                config_updates[key] = data[key]
        
        if config_updates:
            security_system.streaming_manager.set_config(**config_updates)
        
        success = security_system.streaming_manager.start_streaming()
        
        if success and security_system.tts_manager:
            security_system.tts_manager.speak("streaming_started")
        
        return jsonify({
            "success": success,
            "message": "Streaming started" if success else "Streaming already active",
            "config": {
                "fps": security_system.streaming_manager.config.fps,
                "quality": security_system.streaming_manager.config.quality,
                "detection_enabled": security_system.streaming_manager.config.detection_enabled,
                "server_url": security_system.streaming_manager.config.server_url
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/stream/stop', methods=['POST'])
def stop_streaming():
    """Stop video streaming"""
    try:
        success = security_system.streaming_manager.stop_streaming()
        
        if success and security_system.tts_manager:
            security_system.tts_manager.speak("streaming_stopped")
        
        return jsonify({
            "success": success,
            "message": "Streaming stopped",
            "timestamp": datetime.datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/system/restart', methods=['POST'])
def restart_system():
    """Restart system components"""
    try:
        data = request.get_json() or {}
        component = data.get('component', 'all')
        
        results = {}
        
        if component in ['all', 'camera']:
            try:
                if security_system.video_capture:
                    security_system.video_capture.release()
                security_system._setup_camera()
                results['camera'] = 'restarted'
            except Exception as e:
                results['camera'] = f'error: {str(e)}'
        
        if component in ['all', 'tts']:
            try:
                security_system.tts_manager.cleanup()
                security_system.tts_manager = TTSManager()
                security_system.door_lock.tts_manager = security_system.tts_manager
                results['tts'] = 'restarted'
            except Exception as e:
                results['tts'] = f'error: {str(e)}'
        
        if component in ['all', 'streaming']:
            try:
                security_system.streaming_manager.stop_streaming()
                security_system.streaming_manager = StreamingManager(security_system)
                results['streaming'] = 'restarted'
            except Exception as e:
                results['streaming'] = f'error: {str(e)}'
        
        return jsonify({
            "message": f"System component(s) restart attempted",
            "component": component,
            "results": results
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/config', methods=['GET', 'POST'])
def system_config():
    """Get or update system configuration"""
    if request.method == 'GET':
        return jsonify({
            "device_id": security_system.device_id,
            "recognition_cooldown": security_system.recognition_cooldown,
            "known_persons": security_system.known_face_names,
            "backend_url": security_system.backend_api_url,
            "gpio_available": GPIO_AVAILABLE,
            "tts_available": TTS_AVAILABLE
        })
    
    elif request.method == 'POST':
        try:
            data = request.get_json() or {}
            updated = []
            
            if 'recognition_cooldown' in data:
                security_system.recognition_cooldown = max(1, int(data['recognition_cooldown']))
                updated.append('recognition_cooldown')
            
            if 'backend_url' in data:
                security_system.backend_api_url = data['backend_url']
                updated.append('backend_url')
            
            if 'known_persons' in data and isinstance(data['known_persons'], list):
                security_system.known_face_names = data['known_persons']
                updated.append('known_persons')
            
            return jsonify({
                "message": "Configuration updated",
                "updated_fields": updated,
                "current_config": {
                    "device_id": security_system.device_id,
                    "recognition_cooldown": security_system.recognition_cooldown,
                    "known_persons": security_system.known_face_names,
                    "backend_url": security_system.backend_api_url
                }
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

@app.route('/person/add', methods=['POST'])
def add_person():
    """Add a new person to the security system"""
    try:
        data = request.get_json()
        if not data or 'name' not in data:
            return jsonify({"error": "Name is required"}), 400
        
        name = data['name'].strip()
        if not name:
            return jsonify({"error": "Name cannot be empty"}), 400
        
        # Optional: handle image data if provided
        image = None
        if 'image' in data:
            try:
                # Decode base64 image if provided
                import base64
                import numpy as np
                image_data = base64.b64decode(data['image'])
                nparr = np.frombuffer(image_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except Exception as e:
                return jsonify({"error": f"Invalid image data: {str(e)}"}), 400
        
        result = security_system.add_person(name, image)
        
        if result['success']:
            return jsonify(result), 201
        else:
            return jsonify(result), 400
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {signum}, shutting down...")
    security_system.cleanup()
    exit(0)
    
if __name__ == '__main__':
    # Register shutdown handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        logger.info("Starting security system server...")
        app.run(
            host='0.0.0.0',
            port=5001,
            debug=False,  # Set to False for production
            threaded=True,
            use_reloader=False
        )
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
    finally:
        logger.info("Shutting down server...")
        security_system.cleanup()