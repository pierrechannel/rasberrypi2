import time
import logging
import datetime
import cv2
import base64
import numpy as np
from contextlib import contextmanager
from typing import Optional, List, Dict, Any
from tts_manager import TTSManager
from door_lock import DoorLockController
from streaming import StreamingManager
from models import FaceRecognitionResult
from enums import AccessResult

try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    print("WARNING: face_recognition not available. Falling back to mock recognition.")

logger = logging.getLogger(__name__)

class SecuritySystem:
    """Main security system class with enhanced error handling and face recognition"""
    
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
        self.known_face_encodings = []  # Store face encodings for recognition
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
        
        # Initialize known face encodings (mock or load from storage)
        if FACE_RECOGNITION_AVAILABLE:
            self._load_known_faces()

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

    def _load_known_faces(self):
        """Load known face encodings (mock or from storage)"""
        # In a real implementation, load encodings from a persistent storage (e.g., file or database)
        # For now, initialize empty encodings or load pre-trained data
        self.logger.info("Loading known face encodings (mock implementation)")
        # Example: Load encodings if pre-existing images are available
        # This is a placeholder; replace with actual loading logic
        self.known_face_encodings = [None] * len(self.known_face_names)

    def get_frame(self) -> Optional[np.ndarray]:
        """Get current camera frame"""
        if not self.video_capture or not self.video_capture.isOpened():
            return None
        
        try:
            ret, frame = self.video_capture.read()
            return frame if ret else None
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
            self.logger.warning("Using mock face recognition due to missing face_recognition library")
            # Fallback to mock implementation
            height, width = frame.shape[:2]
            face_location = (100, 300, 300, 100)  # (top, right, bottom, left)
            import random
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
            # Convert frame to RGB (face_recognition uses RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                name = "Unknown"
                confidence = 0.0
                access_result = AccessResult.UNKNOWN
                
                # Compare with known faces
                if self.known_face_encodings and any(encoding is not None for encoding in self.known_face_encodings):
                    matches = face_recognition.compare_faces(
                        [enc for enc in self.known_face_encodings if enc is not None],
                        face_encoding,
                        tolerance=0.6
                    )
                    face_distances = face_recognition.face_distance(
                        [enc for enc in self.known_face_encodings if enc is not None],
                        face_encoding
                    )
                    
                    if face_distances.size > 0:
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            name = self.known_face_names[best_match_index]
                            confidence = (1 - face_distances[best_match_index]) * 100  # Convert distance to confidence percentage
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

    def add_person(self, name: str, image: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Add a new person to the known face names and encodings list."""
        try:
            if not name or not isinstance(name, str):
                raise ValueError("Valid name is required")
            
            if name in self.known_face_names:
                raise ValueError(f"Person '{name}' already exists in the system")
            
            # Add name to known_face_names
            self.known_face_names.append(name)
            
            # Initialize encoding as None if no image is provided
            face_encoding = None
            
            # Process image for face encoding if provided
            if image is not None and FACE_RECOGNITION_AVAILABLE:
                try:
                    # Convert to RGB for face_recognition
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    encodings = face_recognition.face_encodings(rgb_image)
                    if encodings:
                        face_encoding = encodings[0]  # Take the first face found
                        self.logger.info(f"Face encoding generated for {name}")
                    else:
                        self.logger.warning(f"No face detected in image for {name}")
                        raise ValueError("No face detected in provided image")
                except Exception as e:
                    self.logger.error(f"Failed to generate face encoding for {name}: {e}")
                    raise ValueError(f"Failed to process image: {str(e)}")
            
            # Add encoding to known_face_encodings
            self.known_face_encodings.append(face_encoding)
            
            # Log the addition
            log_entry = {
                "person_id": f"ID_{hash(name) % 1000}",
                "name": name,
                "timestamp": datetime.datetime.now().isoformat(),
                "action": "person_added",
                "has_encoding": face_encoding is not None
            }
            
            # If an image is provided, store it
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