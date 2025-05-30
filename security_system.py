import time
import logging
import datetime
import cv2
import base64
import numpy as np
from contextlib import contextmanager
from typing import Optional, List, Dict, Any
from threading import Thread, Event, Lock
from tts_manager import TTSManager
from door_lock import DoorLockController
from streaming import StreamingManager
from models import FaceRecognitionResult
from enums import AccessResult
import requests
import io
import threading
import os
from dotenv import load_dotenv
import re

try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    print("WARNING: face_recognition not available. Falling back to mock recognition.")

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class SecuritySystem:
    """Main security system class with continuous face recognition and server sync"""

    def __init__(self, device_id: str = "RPI_001"):
        self.device_id = device_id
        self.logger = logging.getLogger(f"{__name__}.{device_id}")
        
        # Initialize components
        self.tts_manager = TTSManager()
        self.door_lock = DoorLockController(tts_manager=self.tts_manager)
        self.streaming_manager = StreamingManager(self)
        
        # Camera setup
        self.video_capture = None
        self.camera_error_count = 0
        self.max_camera_errors = 5
        self._setup_camera()
        
        # Recognition settings
        self.known_face_names = []
        self.known_face_encodings = []
        self.last_recognition_time = 0
        self.recognition_cooldown = 5
        
        # Thread safety for face data
        self.face_data_lock = Lock()
        
        # Continuous recognition control
        self.recognition_active = True
        self.recognition_thread = None
        self.shutdown_event = Event()
        
        # Server sync settings
        self.server_api_url = os.getenv("SERVER_API_URL", "http://<server-ip>:port/api/persons")
        self.api_key = os.getenv("SERVER_API_KEY", "your-api-key")
        self.sync_interval = 300  # Sync every 5 minutes
        self.sync_thread = None
        self.last_sync_time = 0
        self.synced_person_ids = set()  # Track synced IDs to avoid duplicates
        
        # Backend connection (for local logs)
        self.backend_api_url = "https://example.com/api"
        self.backend_online = True
        self.offline_logs = []
        
        # Startup announcement
        if self.tts_manager:
            self.tts_manager.speak("system_startup")
        
        self.logger.info("Security system initialized")
        
        if FACE_RECOGNITION_AVAILABLE:
            self._load_known_faces()
        
        # Start continuous face recognition and server sync
        self._start_continuous_recognition()
        self._start_server_sync()

    def _setup_camera(self):
        """Initialize camera with error handling and multiple backends"""
        backends = [
            (cv2.CAP_DSHOW, "DirectShow"),  # Prioritize DSHOW to avoid MSMF error
            (cv2.CAP_MSMF, "MSMF"),
            (cv2.CAP_ANY, "Default")
        ]
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            for backend, backend_name in backends:
                try:
                    self.logger.info(f"Attempting to initialize camera with {backend_name} backend (attempt {attempt + 1}/{max_retries})")
                    self.video_capture = cv2.VideoCapture(0, backend)
                    if self.video_capture.isOpened():
                        self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # Lower resolution for Pi
                        self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
                        self.video_capture.set(cv2.CAP_PROP_FPS, 15)
                        self.logger.info(f"Camera initialized successfully with {backend_name} backend")
                        self.camera_error_count = 0
                        return
                    else:
                        self.video_capture.release()
                        self.video_capture = None
                except Exception as e:
                    self.logger.error(f"Camera initialization failed with {backend_name} backend: {e}")
            
            if attempt < max_retries - 1:
                self.logger.info(f"Retrying camera initialization after {retry_delay} seconds")
                time.sleep(retry_delay)
        
        self.logger.error("Failed to initialize camera after all attempts")
        if self.tts_manager:
            self.tts_manager.speak("camera_error")

    def _load_known_faces(self):
        """Load known face encodings from storage"""
        self.logger.info("Loading known face encodings (mock implementation)")
        # Placeholder: Load from file/database in production
        self.known_face_encodings = [None] * len(self.known_face_names)

    def _start_continuous_recognition(self):
        """Start continuous face recognition in a background thread"""
        if self.recognition_thread and self.recognition_thread.is_alive():
            self.logger.info("Continuous recognition already running")
            return
        
        self.recognition_active = True
        self.recognition_thread = Thread(
            target=self._recognition_loop,
            daemon=True,
            name="Face-Recognition-Thread"
        )
        self.recognition_thread.start()
        self.logger.info("Continuous face recognition started")

    def _stop_continuous_recognition(self):
        """Stop continuous face recognition"""
        self.recognition_active = False
        if self.recognition_thread and self.recognition_thread.is_alive():
            self.shutdown_event.set()
            self.recognition_thread.join(timeout=5)
            self.shutdown_event.clear()
        self.logger.info("Continuous face recognition stopped")

    def _recognition_loop(self):
        """Continuous face recognition loop"""
        self.logger.info("Face recognition loop started")
        
        while self.recognition_active and not self.shutdown_event.is_set():
            try:
                frame = self.get_frame()
                if frame is not None:
                    self.process_access_attempt(frame)
                    self.camera_error_count = 0
                else:
                    self.camera_error_count += 1
                    self.logger.warning(f"Failed to capture frame ({self.camera_error_count}/{self.max_camera_errors})")
                    if self.camera_error_count >= self.max_camera_errors:
                        self.logger.error("Too many camera errors, attempting to reinitialize camera")
                        if self.video_capture:
                            self.video_capture.release()
                        self._setup_camera()
                        self.camera_error_count = 0
                        if self.tts_manager:
                            self.tts_manager.speak("camera_error")
                
                time.sleep(1.0)  # 1 FPS for Raspberry Pi performance
            except Exception as e:
                self.logger.error(f"Recognition loop error: {e}")
                time.sleep(1)

    def _start_server_sync(self):
        """Start background thread for server synchronization"""
        if self.sync_thread and self.sync_thread.is_alive():
            self.logger.info("Server sync already running")
            return
        
        self.sync_thread = Thread(
            target=self._sync_loop,
            daemon=True,
            name="Server-Sync-Thread"
        )
        self.sync_thread.start()
        self.logger.info("Server sync thread started")

    def _sync_loop(self):
        """Periodically sync with server"""
        self.logger.info("Server sync loop started")
        
        while not self.shutdown_event.is_set():
            try:
                current_time = time.time()
                if (current_time - self.last_sync_time) >= self.sync_interval:
                    self.sync_with_server()
                    self.last_sync_time = current_time
                
                time.sleep(60)
            except Exception as e:
                self.logger.error(f"Server sync loop error: {e}")
                time.sleep(60)

    def sync_with_server(self):
        """Fetch person data from server and update face recognition database"""
        self.logger.info("Attempting to sync with server")
        
        max_retries = 3
        retry_delay = 5
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        for attempt in range(max_retries):
            try:
                response = requests.get(self.server_api_url, headers=headers, timeout=10)
                response.raise_for_status()
                persons = response.json()
                
                new_names = []
                new_encodings = []
                new_person_ids = []
                
                for person in persons:
                    person_id = person.get("id")
                    name = person.get("name")
                    image_data = person.get("image")
                    
                    if not all([person_id, name, image_data]):
                        self.logger.warning(f"Skipping person with missing data: {person}")
                        continue
                    
                    if person_id in self.synced_person_ids:
                        self.logger.info(f"Person ID {person_id} ({name}) already synced, skipping")
                        continue
                    
                    # Validate base64 data
                    if not re.match(r'^[A-Za-z0-9+/=]+$', image_data):
                        self.logger.error(f"Invalid base64 data for {name}")
                        continue
                    
                    try:
                        image_bytes = base64.b64decode(image_data)
                        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
                        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                        
                        if image is None:
                            self.logger.error(f"Failed to decode image for {name}")
                            continue
                        
                        # Resize image for performance
                        if image.shape[0] > 240:
                            scale = 240 / image.shape[0]
                            image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
                        
                        if FACE_RECOGNITION_AVAILABLE:
                            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            encodings = face_recognition.face_encodings(rgb_image)
                            if encodings:
                                new_names.append(name)
                                new_encodings.append(encodings[0])
                                new_person_ids.append(person_id)
                                self.logger.info(f"Generated face encoding for {name}")
                            else:
                                self.logger.warning(f"No face detected in image for {name}")
                        else:
                            new_names.append(name)
                            new_encodings.append(None)
                            new_person_ids.append(person_id)
                            self.logger.info(f"Added {name} without encoding (mock mode)")
                            
                    except Exception as e:
                        self.logger.error(f"Error processing image for {name}: {e}")
                        continue
                
                # Update face data safely
                with self.face_data_lock:
                    self.known_face_names.extend(new_names)
                    self.known_face_encodings.extend(new_encodings)
                    self.synced_person_ids.update(new_person_ids)
                
                self.logger.info(f"Synced {len(new_names)} new persons from server")
                if self.tts_manager and new_names:
                    self.tts_manager.speak_custom(f"Synced {len(new_names)} new persons", priority=True)
                
                self.offline_logs.append({
                    "event": "server_sync",
                    "new_persons": len(new_names),
                    "timestamp": datetime.datetime.now().isoformat()
                })
                
                return
            
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Server sync attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (2 ** attempt))
                else:
                    self.logger.error("Max retries reached for server sync")
                    if self.tts_manager:
                        self.tts_manager.speak("server_sync_failed")

    def get_frame(self) -> Optional[np.ndarray]:
        """Get current camera frame"""
        if not self.video_capture or not self.video_capture.isOpened():
            self.logger.warning("Camera not initialized or not opened")
            return None
        
        try:
            ret, frame = self.video_capture.read()
            if not ret:
                self.logger.warning("Failed to grab frame")
                return None
            return frame
        except Exception as e:
            self.logger.error(f"Frame capture error: {e}")
            return None

    def process_face_recognition(self, frame: np.ndarray) -> List[FaceRecognitionResult]:
        """Process face recognition on frame using face_recognition library"""
        results = []
        
        if frame is None:
            self.logger.warning("No frame provided for face recognition")
            return results

        if not FACE_RECOGNITION_AVAILABLE:
            self.logger.warning("Using mock face recognition")
            height, width = frame.shape[:2]
            face_location = (100, 300, 300, 100)
            import random
            with self.face_data_lock:
                names = self.known_face_names + ["Unknown"]
            name = random.choice(names)
            confidence = random.uniform(75, 95) if name != "Unknown" else random.uniform(30, 60)
            access_result = AccessResult.GRANTED if name in self.known_face_names else AccessResult.UNKNOWN
            
            result = FaceRecognitionResult(
                person_id=f"ID_{hash(name) % 1000}",
                name=name,
                confidence=confidence,
                location=face_location,
                access_result=access_result
            )
            results.append(result)
            return results

        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
            with self.face_data_lock:
                known_encodings = [enc for enc in self.known_face_encodings if enc is not None]
                known_names = self.known_face_names.copy()
            
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                name = "Unknown"
                confidence = 0.0
                access_result = AccessResult.UNKNOWN
                
                if known_encodings:
                    matches = face_recognition.compare_faces(
                        known_encodings,
                        face_encoding,
                        tolerance=0.6
                    )
                    face_distances = face_recognition.face_distance(
                        known_encodings,
                        face_encoding
                    )
                    
                    if face_distances.size > 0:
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            name = known_names[best_match_index]
                            confidence = (1 - face_distances[best_match_index]) * 100
                            access_result = AccessResult.GRANTED
                
                result = FaceRecognitionResult(
                    person_id=f"ID_{hash(name) % 1000}",
                    name=name,
                    confidence=confidence,
                    location=(top, right, bottom, left),
                    access_result=access_result
                )
                results.append(result)
                
                self.logger.info(f"Detected {name} with confidence {confidence:.1f}%")
        
        except Exception as e:
            self.logger.error(f"Face recognition error: {e}")
        
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

    def process_access_attempt(self, frame: np.ndarray) -> Dict[str, Any]:
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
                    if result.access_result == AccessResult.GRANTED:
                        self.door_lock.unlock_door(duration=5, name=result.name)
                    elif result.access_result == AccessResult.UNKNOWN:
                        self.door_lock.handle_unknown_person()
                    else:
                        self.door_lock.handle_access_denied()
                    
                    self._log_access_attempt(result, frame)
                    
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

    def _log_access_attempt(self, result: FaceRecognitionResult, frame: np.ndarray):
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

    def _image_to_base64(self, frame: np.ndarray) -> str:
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
            if self.offline_logs:
                self.logger.info(f"Syncing {len(self.offline_logs)} logs with backend")
                self.offline_logs.clear()
                if self.tts_manager:
                    self.tts_manager.speak("sync_successful")
            return True
        except Exception as e:
            self.logger.error(f"Backend sync failed: {e}")
            if self.tts_manager:
                self.tts_manager.speak("backend_offline")
            return False

    def add_person(self, name: str, image: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Add a new person to the known face names and encodings list"""
        try:
            if not name or not isinstance(name, str):
                raise ValueError("Valid name is required")
            
            with self.face_data_lock:
                if name in self.known_face_names:
                    raise ValueError(f"Person '{name}' already exists in the system")
                
                self.known_face_names.append(name)
                
                face_encoding = None
                if image is not None and FACE_RECOGNITION_AVAILABLE:
                    try:
                        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        encodings = face_recognition.face_encodings(rgb_image)
                        if encodings:
                            face_encoding = encodings[0]
                            self.logger.info(f"Face encoding generated for {name}")
                        else:
                            self.logger.warning(f"No face detected in image for {name}")
                            raise ValueError("No face detected in provided image")
                    except Exception as e:
                        self.logger.error(f"Failed to generate face encoding for {name}: {e}")
                        raise ValueError(f"Failed to process image: {str(e)}")
                
                self.known_face_encodings.append(face_encoding)
            
            log_entry = {
                "person_id": f"ID_{hash(name) % 1000}",
                "name": name,
                "timestamp": datetime.datetime.now().isoformat(),
                "action": "person_added",
                "has_encoding": face_encoding is not None
            }
            
            if image is not None:
                log_entry["image"] = self._image_to_base64(image)
            
            self.offline_logs.append(log_entry)
            self.logger.info(f"Added new person: {name}")
            
            if self.tts_manager:
                self.tts_manager.speak_custom(f"New person {name} added to the system", priority=True)
            
            self.sync_with_backend()
            
            return {
                "success": True,
                "message": f"Person '{name}' added successfully",
                "person_id": log_entry["person_id"],
                "known_persons_count": len(self.known_face_names),
                "has_encoding": face_encoding is not None
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
            time.sleep(2)
        
        self._stop_continuous_recognition()
        self.shutdown_event.set()
        if self.sync_thread and self.sync_thread.is_alive():
            self.sync_thread.join(timeout=5)
        
        self.streaming_manager.stop_streaming()
        
        if self.video_capture:
            self.video_capture.release()
        
        self.door_lock.cleanup()
        
        if self.tts_manager:
            self.tts_manager.cleanup()
        
        self.logger.info("Security system shutdown complete")