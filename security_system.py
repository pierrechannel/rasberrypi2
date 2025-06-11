
import time
import logging
import datetime
import cv2
import requests
import os
import numpy as np
from typing import Any, Optional, List, Dict, Tuple
from threading import Thread, Event, Lock
from contextlib import contextmanager
from collections import deque
import random
import pickle
import tempfile
import json

try:
    from deepface import DeepFace
    import tensorflow as tf
    # Suppress TensorFlow warnings
    tf.get_logger().setLevel('ERROR')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    DEEPFACE_AVAILABLE = True
    print("DeepFace initialized successfully")
except ImportError:
    DEEPFACE_AVAILABLE = False
    print("WARNING: DeepFace not available. Falling back to mock recognition.")

from image_processor import EnhancedImageProcessor
from data_poster import RobustDataPoster
from tts_manager import TTSManager
from door_lock import DoorLockController
from streaming import StreamingManager
from models import FaceRecognitionResult
from enums import AccessResult
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class SecuritySystem:
    """Enhanced security system with DeepFace integration - No offline storage"""

    def __init__(self, device_id: str = "RPI_001"):
        self.device_id = device_id
        self.logger = logging.getLogger(f"{__name__}.{device_id}")
        self.image_processor = EnhancedImageProcessor()
        self.data_poster = RobustDataPoster()
        self.tts_manager = TTSManager()
        self.door_lock = DoorLockController()
        self.streaming_manager = StreamingManager(self)
        
        # Camera setup
        self.video_access = None
        self.camera_error_count = 0
        self.max_camera_errors = 5
        self._setup_enhanced_camera()
        
        # DeepFace configuration
        self.deepface_model = "Facenet512"  # Options: VGG-Face, Facenet, Facenet512, OpenFace, DeepFace, DeepID, ArcFace, Dlib, SFace
        self.deepface_detector = "opencv"   # Options: opencv, ssd, dlib, mtcnn, retinaface, mediapipe
        self.distance_metric = "cosine"     # Options: cosine, euclidean, euclidean_l2
        
        # Face data storage
        self.known_face_data = {}  # {name: {"embedding": array, "metadata": dict}}
        self.known_face_ids = {}
        self.face_encoding_quality_scores = {}
        
        # Recognition settings
        self.last_recognition_time = 0
        self.recognition_cooldown = 10
        self.recognition_history = deque(maxlen=5)
        self.confidence_threshold = 0.7
        self.face_data_lock = Lock()
        self.recognition_active = True
        self.recognition_thread = None
        self.shutdown_event = Event()
        
        # Server configuration
        self.server_users_url = os.getenv("SERVER_USERS_URL", "https://apps.mediabox.bi:26875/administration/warehouse_users")
        self.access_log_url = os.getenv("SERVER_ACCESS_LOG_URL", "https://apps.mediabox.bi:26875/warehouse_acces/create")
        self.sync_interval = 180
        self.sync_thread = None
        self.last_sync_time = 0
        self.synced_person_ids = set()
        
        # Distance thresholds for different models
        self.distance_thresholds = {
            "VGG-Face": {"strict": 0.4, "normal": 0.6},
            "Facenet": {"strict": 0.4, "normal": 0.6},
            "Facenet512": {"strict": 0.3, "normal": 0.4},
            "OpenFace": {"strict": 0.4, "normal": 0.6},
            "DeepFace": {"strict": 0.35, "normal": 0.55},
            "DeepID": {"strict": 0.4, "normal": 0.6},
            "ArcFace": {"strict": 0.4, "normal": 0.6},
            "Dlib": {"strict": 0.4, "normal": 0.6},
            "SFace": {"strict": 0.4, "normal": 0.6}
        }
        
        # Initialize system
        if self.tts_manager:
            self.tts_manager.speak("system_startup")
        self.logger.info(f"Enhanced security system initialized with DeepFace model: {self.deepface_model}")
        
        if DEEPFACE_AVAILABLE:
            self._load_known_faces_enhanced()
        self._start_continuous_recognition()
        self._start_server_sync()

    def _setup_enhanced_camera(self):
        """Initialize camera with enhanced settings for better image quality"""
        backends = [
            (cv2.CAP_DSHOW, "DirectShow"),
            (cv2.CAP_MSMF, "MSMF"),
            (cv2.CAP_V4L2, "V4L2"),
            (cv2.CAP_ANY, "Default")
        ]
        
        for backend, backend_name in backends:
            try:
                self.logger.info(f"Initializing camera with {backend_name} backend")
                self.video_access = cv2.VideoCapture(0, backend)
                
                if self.video_access.isOpened():
                    # Set camera properties
                    self.video_access.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.video_access.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    self.video_access.set(cv2.CAP_PROP_FPS, 30)
                    self.video_access.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)
                    self.video_access.set(cv2.CAP_PROP_CONTRAST, 0.5)
                    self.video_access.set(cv2.CAP_PROP_SATURATION, 0.5)
                    self.video_access.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
                    self.video_access.set(cv2.CAP_PROP_AUTOFOCUS, 1)
                    
                    # Test frame capture
                    ret, test_frame = self.video_access.read()
                    if ret and test_frame is not None:
                        self.logger.info(f"Camera initialized successfully with {backend_name}")
                        self.camera_error_count = 0
                        return
                    else:
                        self.video_access.release()
            except Exception as e:
                self.logger.error(f"Camera initialization failed with {backend_name}: {e}")
                continue
        
        self.logger.error("Failed to initialize camera")
        if self.tts_manager:
            self.tts_manager.speak("camera_error")

    def _load_known_faces_enhanced(self):
        """Load known faces with enhanced DeepFace processing"""
        self.logger.info(f"Loading known face embeddings with DeepFace ({self.deepface_model})")
        self.sync_with_server()

    def _extract_face_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Extract face embedding using DeepFace"""
        if not DEEPFACE_AVAILABLE:
            return None
        
        try:
            # Save image temporarily for DeepFace processing
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                cv2.imwrite(tmp_file.name, image)
                temp_path = tmp_file.name
            
            try:
                # Extract embedding using DeepFace
                embedding_objs = DeepFace.represent(
                    img_path=temp_path,
                    model_name=self.deepface_model,
                    detector_backend=self.deepface_detector,
                    enforce_detection=True
                )
                
                if embedding_objs and len(embedding_objs) > 0:
                    embedding = np.array(embedding_objs[0]["embedding"])
                    return embedding
                else:
                    return None
                    
            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    
        except Exception as e:
            self.logger.error(f"DeepFace embedding extraction error: {e}")
            return None

    def _detect_faces_deepface(self, image: np.ndarray) -> List[Dict]:
        """Detect faces using DeepFace"""
        if not DEEPFACE_AVAILABLE:
            return []
        
        try:
            # Save image temporarily for DeepFace processing
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                cv2.imwrite(tmp_file.name, image)
                temp_path = tmp_file.name
            
            try:
                # Detect faces
                face_objs = DeepFace.extract_faces(
                    img_path=temp_path,
                    detector_backend=self.deepface_detector,
                    enforce_detection=False,
                    align=True
                )
                
                faces = []
                for face_obj in enumerate(face_objs):
                    if face_obj["face"] is not None:
                        # Convert normalized face back to original scale
                        face_array = (face_obj["face"] * 255).astype(np.uint8)
                        facial_area = face_obj["facial_area"]
                        faces.append({
                            "face": face_array,
                            "region": {
                                "x": facial_area["x"],
                                "y": facial_area["y"],
                                "w": facial_area["w"],
                                "h": facial_area["h"]
                            },
                            "confidence": face_obj.get("confidence", 0.9)
                        })
                
                return faces
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    
        except Exception as e:
            self.logger.error(f"DeepFace face detection error: {e}")
            return []

    def process_face_recognition_enhanced(self, frame: np.ndarray, bypass_cooldown: bool = False) -> List[FaceRecognitionResult]:
        """Enhanced face recognition using DeepFace"""
        results = []
        if frame is None:
            return results
        
        try:
            if not bypass_cooldown:
                with self.recognition_cooldown_check():
                    results = self._process_frame_with_deepface(frame)
            else:
                results = self._process_frame_with_deepface(frame)
                
        except Exception as e:
            self.logger.error(f"Enhanced DeepFace recognition error: {e}")
        
        return results

    def _process_frame_with_deepface(self, frame: np.ndarray) -> List[FaceRecognitionResult]:
        """Process frame with DeepFace recognition"""
        results = []
        
        try:
            # Enhance image quality
            enhanced_frame = self.image_processor.enhance_image_quality(frame)
            
            # Detect faces using DeepFace
            detected_faces = self._detect_faces_deepface(enhanced_frame)
            
            for face_data in detected_faces:
                face_image = face_data["face"]
                face_region = face_data["region"]
                
                # Extract embedding
                embedding = self._extract_face_embedding(face_image)
                if embedding is None:
                    continue
                
                # Match against known faces
                name, confidence, access_result = self._match_face_embedding(embedding)
                
                # Convert region to face_recognition format (top, right, bottom, left)
                face_location = (
                    face_region["y"],
                    face_region["x"] + face_region["w"],
                    face_region["y"] + face_region["h"],
                    face_region["x"]
                )
                
                result = FaceRecognitionResult(
                    person_id=self.known_face_ids.get(name, f"ID_{hash(name) % 1000}"),
                    name=name,
                    confidence=confidence,
                    location=face_location,
                    access_result=access_result
                )
                results.append(result)
                self.logger.info(f"DeepFace recognition: {name} with confidence {confidence:.2f}")
        
        except Exception as e:
            self.logger.error(f"DeepFace frame processing error: {e}")
        
        return results

    def _match_face_embedding(self, embedding: np.ndarray) -> Tuple[str, float, AccessResult]:
        """Match face embedding against known faces using DeepFace distance metrics"""
        if not DEEPFACE_AVAILABLE:
            return "Unknown", 0.0, AccessResult.UNKNOWN
        
        with self.face_data_lock:
            if not self.known_face_data:
                return "Unknown", 0.0, AccessResult.UNKNOWN
        
        try:
            best_match_name = "Unknown"
            best_distance = float('inf')
            best_confidence = 0.0
            
            # Get thresholds for current model
            thresholds = self.distance_thresholds.get(self.deepface_model, {"strict": 0.4, "normal": 0.6})
            
            with self.face_data_lock:
                for name, face_data in self.known_face_data.items():
                    known_embedding = face_data["embedding"]
                    
                    # Calculate distance using specified metric
                    if self.distance_metric == "cosine":
                        distance = 1 - np.dot(embedding, known_embedding) / (
                            np.linalg.norm(embedding) * np.linalg.norm(known_embedding)
                        )
                    elif self.distance_metric == "euclidean":
                        distance = np.linalg.norm(embedding - known_embedding)
                    elif self.distance_metric == "euclidean_l2":
                        distance = np.linalg.norm(embedding - known_embedding) / len(embedding)
                    else:
                        distance = 1 - np.dot(embedding, known_embedding) / (
                            np.linalg.norm(embedding) * np.linalg.norm(known_embedding)
                        )
                    
                    if distance < best_distance:
                        best_distance = distance
                        best_match_name = name
            
            # Convert distance to confidence score (0-100)
            if self.distance_metric == "cosine":
                confidence = max(0, (1 - best_distance) * 100)
            else:
                confidence = max(0, (1 - (best_distance / 2)) * 100)
            
            # Determine access result based on thresholds
            if best_distance <= thresholds["strict"]:
                access_result = AccessResult.GRANTED
            elif best_distance <= thresholds["normal"]:
                if self._validate_with_history(best_match_name, confidence):
                    access_result = AccessResult.GRANTED
                else:
                    access_result = AccessResult.UNKNOWN
            else:
                best_match_name = "Unknown"
                access_result = AccessResult.UNKNOWN
            
            return best_match_name, confidence, access_result
        
        except Exception as e:
            self.logger.error(f"DeepFace matching error: {e}")
            return "Unknown", 0.0, AccessResult.UNKNOWN

    def _validate_with_history(self, name: str, confidence: float) -> bool:
        """Validate recognition using historical data"""
        self.recognition_history.append({
            "name": name, 
            "confidence": confidence, 
            "timestamp": time.time()
        })
        
        recent_recognitions = [
            r for r in self.recognition_history 
            if time.time() - r["timestamp"] < 30
        ]
        
        same_person_count = sum(1 for r in recent_recognitions if r["name"] == name)
        avg_confidence = np.mean([
            r["confidence"] for r in recent_recognitions if r["name"] == name
        ]) if recent_recognitions else 0.0
        
        return same_person_count >= 2 and avg_confidence >= 65.0

    def _log_access_attempt(self, result: FaceRecognitionResult, frame: np.ndarray):
        """Log access attempt without offline storage"""
        try:
            timestamp = datetime.datetime.now().isoformat() + "Z"
            
            with self.face_data_lock:
                warehouse_user_id = self.known_face_ids.get(result.name, "")
            
            enhanced_frame = self.image_processor.enhance_image_quality(frame)
            _, buffer = cv2.imencode('.jpg', enhanced_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            access_data = {
                "WAREHOUSE_USER_ID": warehouse_user_id,
                "STATUT": 1 if result.access_result == AccessResult.GRANTED else 2,
                "DATE_SAVE": timestamp,
                "DEVICE_ID": self.device_id,
                "CONFIDENCE": round(result.confidence, 2),
                "RECOGNITION_METHOD": f"DeepFace_{self.deepface_model}",
                "DISTANCE_METRIC": self.distance_metric,
                "FACE_LOCATION": {
                    "top": result.location[0],
                    "right": result.location[1], 
                    "bottom": result.location[2],
                    "left": result.location[3]
                }
            }
            
            files = {"IMAGE": ("access.jpg", buffer.tobytes(), "image/jpeg")}
            
            success, response = self.data_poster.post_with_exponential_backoff(
                self.access_log_url, data=access_data, files=files
            )
            
            if success:
                self.logger.info(f"Access logged successfully for {result.name}")
                return
            
            success, response = self.data_poster.post_with_exponential_backoff(
                self.access_log_url, json_data=access_data
            )
            
            if success:
                self.logger.info(f"Access logged (no image) for {result.name}")
                return
            
            self.logger.error(f"Failed to log access attempt for {result.name}: {response}")
            if self.tts_manager:
                self.tts_manager.speak("logging_failed")
                
        except Exception as e:
            self.logger.error(f"Access logging failed: {e}")

    def process_access_attempt(self, frame: np.ndarray) -> Dict[str, Any]:
        """Enhanced access attempt processing with DeepFace"""
        try:
            with self.recognition_cooldown_check():
                results = self.process_face_recognition_enhanced(frame)
                
                response_data = {
                    "faces_detected": len(results),
                    "results": [],
                    "timestamp": datetime.datetime.now().isoformat(),
                    "device_id": self.device_id,
                    "processing_enhanced": True,
                    "deepface_model": self.deepface_model,
                    "distance_metric": self.distance_metric
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
                        "confidence": round(result.confidence, 2),
                        "enhanced_processing": True,
                        "deepface_model": self.deepface_model
                    })
                
                return response_data
                
        except Exception as e:
            self.logger.error(f"Enhanced access processing error: {e}")
            raise

    def sync_with_server(self):
        """Enhanced server synchronization with DeepFace processing"""
        self.logger.info("Starting enhanced server sync with DeepFace processing")
        
        success, response = self.data_poster.get_with_exponential_backoff(
            self.server_users_url
        )
        
        if not success:
            self.logger.error(f"Server sync failed: {response}")
            if self.tts_manager:
                self.tts_manager.speak("server_sync_failed")
            return
        
        try:
            data = response if isinstance(response, dict) else {}
            
            if data.get("statusCode") != 200:
                self.logger.error(f"API returned status: {data.get('statusCode')}")
                return
            
            persons = data.get("result", {}).get("data", [])
            self.logger.info(f"Retrieved {len(persons)} persons from server")
            
            new_count = self._process_server_persons(persons)
            
            if new_count > 0:
                self.logger.info(f"DeepFace sync: Added {new_count} new persons")
                if self.tts_manager:
                    self.tts_manager.speak_custom(f"DeepFace sync complete: {new_count} new persons")
                    
        except Exception as e:
            self.logger.error(f"Enhanced server sync processing error: {e}")

    def _process_server_persons(self, persons: List[Dict]) -> int:
        """Process persons from server with DeepFace embedding generation"""
        new_count = 0
        
        for person in persons:
            try:
                person_id = str(person.get("WAREHOUSE_USER_ID", ""))
                nom = person.get("NOM", "").strip()
                prenom = person.get("PRENOM", "").strip()
                photo_url = person.get("PHOTO", "")
                
                if not all([person_id, nom, prenom, photo_url]):
                    continue
                
                if person_id in self.synced_person_ids:
                    continue
                
                name = f"{nom} {prenom}"
                
                image = self._download_and_process_image(photo_url)
                if image is None:
                    continue
                
                embedding = self._extract_face_embedding(image)
                if embedding is None and DEEPFACE_AVAILABLE:
                    self.logger.warning(f"Failed to generate embedding for {name}")
                    continue
                
                with self.face_data_lock:
                    self.known_face_data[name] = {
                        "embedding": embedding,
                        "metadata": {
                            "person_id": person_id,
                            "nom": nom,
                            "prenom": prenom,
                            "sync_timestamp": time.time(),
                            "model": self.deepface_model
                        }
                    }
                    self.known_face_ids[name] = person_id
                    self.synced_person_ids.add(person_id)
                
                new_count += 1
                self.logger.info(f"DeepFace processing: Added {name}")
                
            except Exception as e:
                self.logger.error(f"Error processing person with DeepFace: {e}")
                continue
        
        return new_count

    def _download_and_process_image(self, photo_url: str) -> Optional[np.ndarray]:
        """Download and enhance image for DeepFace processing"""
        try:
            photo_url = photo_url.replace("\\", "/")
            response = requests.get(photo_url, timeout=15)
            response.raise_for_status()
            
            image_array = np.frombuffer(response.content, dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if image is None:
                return None
            
            enhanced_image = self.image_processor.enhance_image_quality(image)
            
            if enhanced_image.shape[0] > 480:
                scale = 480 / enhanced_image.shape[0]
                enhanced_image = cv2.resize(enhanced_image, (0, 0), fx=scale, fy=scale)
            
            return enhanced_image
            
        except Exception as e:
            self.logger.error(f"Image download/processing error: {e}")
            return None

    def _mock_recognition(self, frame: np.ndarray) -> List[FaceRecognitionResult]:
        """Enhanced mock recognition for testing when DeepFace is not available"""
        results = []
        height, width = frame.shape[:2]
        
        face_location = (
            random.randint(50, height//3),
            random.randint(2*width//3, width-50),
            random.randint(2*height//3, height-50),
            random.randint(50, width//3)
        )
        
        with self.face_data_lock:
            names = list(self.known_face_data.keys()) + ["Unknown"]
        
        name = random.choice(names) if names else "Unknown"
        
        if name != "Unknown" and name in self.known_face_data:
            confidence = random.uniform(75, 95)
            access_result = AccessResult.GRANTED
        else:
            confidence = random.uniform(30, 60)
            access_result = AccessResult.UNKNOWN
        
        result = FaceRecognitionResult(
            person_id=self.known_face_ids.get(name, f"ID_{hash(name) % 1000}"),
            name=name,
            confidence=confidence,
            location=face_location,
            access_result=access_result
        )
        results.append(result)
        
        return results

    def _recognition_loop(self):
        """Enhanced continuous face recognition loop with DeepFace"""
        self.logger.info("Enhanced DeepFace recognition loop started")
        frame_skip_counter = 0
        process_every_n_frames = 3
        
        while self.recognition_active and not self.shutdown_event.is_set():
            try:
                frame = self.get_frame()
                if frame is not None:
                    frame_skip_counter += 1
                    if frame_skip_counter >= process_every_n_frames:
                        frame_skip_counter = 0
                        if DEEPFACE_AVAILABLE:
                            self.process_access_attempt(frame)
                        else:
                            results = self._mock_recognition(frame)
                            for result in results:
                                self._log_access_attempt(result, frame)
                    
                    self.camera_error_count = 0
                else:
                    self.camera_error_count += 1
                    self.logger.warning(f"Failed to capture frame ({self.camera_error_count}/{self.max_camera_errors})")
                    
                    if self.camera_error_count >= self.max_camera_errors:
                        self.logger.error("Too many camera errors, reinitializing camera")
                        if self.video_access:
                            self.video_access.release()
                        self._setup_enhanced_camera()
                        self.camera_error_count = 0
                        if self.tts_manager:
                            self.tts_manager.speak("camera_error")
                
                time.sleep(0.1 if frame is not None else 1.0)
                
            except Exception as e:
                self.logger.error(f"Enhanced DeepFace recognition loop error: {e}")
                time.sleep(1)

    def get_frame_enhanced(self) -> Optional[np.ndarray]:
        """Get camera frame with enhanced error handling and quality checks"""
        if not self.video_access or not self.video_access.isOpened():
            return None
        
        try:
            best_frame = None
            best_score = 0
            
            for attempt in range(3):
                ret, frame = self.video_access.read()
                if not ret or frame is None:
                    continue
                
                quality_score = self._calculate_frame_quality(frame)
                if quality_score > best_score:
                    best_score = quality_score
                    best_frame = frame.copy()
                
                if quality_score > 0.8:
                    break
            
            return best_frame
            
        except Exception as e:
            self.logger.error(f"Enhanced frame capture error: {e}")
            return None

    def _calculate_frame_quality(self, frame: np.ndarray) -> float:
        """Calculate frame quality score for selection"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_score = min(laplacian_var / 1000.0, 1.0)
            
            brightness_score = 1.0 - abs(np.mean(gray) - 128) / 128.0
            
            contrast_score = gray.std() / 128.0
            contrast_score = min(contrast_score, 1.0)
            
            quality_score = (sharpness_score * 0.5 + brightness_score * 0.3 + contrast_score * 0.2)
            
            return quality_score
            
        except Exception as e:
            self.logger.error(f"Frame quality calculation error: {e}")
            return 0.0

    def get_frame(self) -> Optional[np.ndarray]:
        """Get current camera frame (enhanced version)"""
        return self.get_frame_enhanced()

    @contextmanager
    def recognition_cooldown_check(self):
        """Enhanced cooldown management with adaptive timing"""
        current_time = time.time()
        time_since_last = current_time - self.last_recognition_time
        
        recent_activity = len([
            r for r in self.recognition_history 
            if current_time - r.get("timestamp", 0) < 60
        ])
        
        adaptive_cooldown = max(5, self.recognition_cooldown - (recent_activity * 2))
        
        if time_since_last < adaptive_cooldown:
            remaining = adaptive_cooldown - time_since_last
            self.logger.info(f"Recognition cooldown: {remaining:.1f}s remaining")
            if self.tts_manager and remaining > 5:
                self.tts_manager.speak("recognition_cooldown")
            raise Exception(f"Recognition cooldown active. {remaining:.1f}s remaining")
        
        yield
        self.last_recognition_time = current_time

    def add_person_enhanced(self, name: str, image: Optional[np.ndarray] = None, 
                           metadata: Dict = None) -> Dict[str, Any]:
        """Enhanced person addition with DeepFace processing"""
        try:
            if not name or not isinstance(name, str):
                raise ValueError("Valid name is required")
            name = name.strip()
            if len(name) < 2:
                raise ValueError("Name must be at least 2 characters")
            
            with self.face_data_lock:
                if name in self.known_face_data:
                    raise ValueError(f"{name} already exists")
                
                face_embedding = None
                quality_score = 0.0
                
                if image is not None:
                    enhanced_image = self.image_processor.enhance_image_quality(image)
                    quality_score = self._calculate_frame_quality(enhanced_image)
                    
                    if quality_score < 0.3:
                        raise ValueError("Image quality too low for reliable recognition")
                    
                    if DEEPFACE_AVAILABLE:
                        face_embedding = self._extract_face_embedding(enhanced_image)
                        if face_embedding is None:
                            raise ValueError("No face detected or embedding failed")
                    
                    self.logger.info(f"Person {name} added with quality score: {quality_score:.2f}")
                
                person_id = f"LOCAL_{hash(name + str(time.time())) % 10000}"
                
                self.known_face_data[name] = {
                    "embedding": face_embedding,
                    "metadata": metadata or {
                        "person_id": person_id,
                        "add_timestamp": time.time(),
                        "model": self.deepface_model if DEEPFACE_AVAILABLE else "mock"
                    }
                }
                self.known_face_ids[name] = person_id
                self.face_encoding_quality_scores[name] = quality_score
            
            log_entry = {
                "person_id": person_id,
                "name": name,
                "timestamp": datetime.datetime.now().isoformat(),
                "action": "person_added_enhanced",
                "has_embedding": face_embedding is not None,
                "quality_score": quality_score,
                "metadata": metadata or {},
                "deepface_model": self.deepface_model if DEEPFACE_AVAILABLE else "mock"
            }
            
            if image is not None:
                _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                files = {"IMAGE": ("person.jpg", buffer.tobytes(), "image/jpeg")}
                success, response = self.data_poster.post_with_exponential_backoff(
                    self.access_log_url, data=log_entry, files=files
                )
                if not success:
                    self.logger.error(f"Failed to log person addition: {response}")
            
            self.logger.info(f"Enhanced person addition: {name} (Quality: {quality_score:.2f})")
            if self.tts_manager:
                self.tts_manager.speak_custom(
                    f"New person {name} added with {quality_score*100:.0f}% quality rating",
                    priority=True
                )
            
            return {
                "success": True,
                "message": f"Person '{name}' added successfully with enhanced processing",
                "person_id": person_id,
                "quality_score": quality_score,
                "known_persons_count": len(self.known_face_data),
                "has_embedding": face_embedding is not None,
                "enhanced_processing": True,
                "deepface_model": self.deepface_model if DEEPFACE_AVAILABLE else "mock"
            }
        
        except Exception as e:
            self.logger.error(f"Enhanced person addition failed: {e}")
            return {
                "success": False,
                "message": f"Failed to add person: {str(e)}",
                "enhanced_processing": True
            }

    def get_system_status_enhanced(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        with self.face_data_lock:
            known_count = len(self.known_face_data)
            avg_quality = np.mean(list(self.face_encoding_quality_scores.values())) if self.face_encoding_quality_scores else 0.0
        
        return {
            "device_id": self.device_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "camera_status": "active" if (self.video_access and self.video_access.isOpened()) else "error",
            "recognition_active": self.recognition_active,
            "deepface_available": DEEPFACE_AVAILABLE,
            "deepface_model": self.deepface_model,
            "distance_metric": self.distance_metric,
            "known_persons": known_count,
            "average_encoding_quality": round(avg_quality, 2),
            "synced_persons": len(self.synced_person_ids),
            "recognition_history_size": len(self.recognition_history),
            "last_sync_time": datetime.datetime.fromtimestamp(self.last_sync_time).isoformat() if self.last_sync_time else None,
            "enhancement_features": {
                "image_enhancement": True,
                "face_detection": DEEPFACE_AVAILABLE,
                "multi_frame_validation": True,
                "adaptive_cooldown": True,
                "quality_scoring": True,
                "robust_data_posting": True
            },
            "error_counts": {
                "camera_errors": self.camera_error_count,
                "max_camera_errors": self.max_camera_errors
            }
        }

    def cleanup_enhanced(self):
        """Enhanced cleanup with comprehensive resource management"""
        self.logger.info("Shutting down enhanced security system")
        
        if self.tts_manager:
            self.tts_manager.speak("system_shutdown")
            time.sleep(1)
        
        self._stop_continuous_recognition()
        self.shutdown_event.set()
        
        if self.sync_thread and self.sync_thread.is_alive():
            self.logger.info("Stopping Server-Sync thread")
            self.sync_thread.join(timeout=3)
            if self.sync_thread.is_alive():
                self.logger.warning("Server-Sync thread did not stop gracefully")
        
        self.streaming_manager.stop_streaming()
        
        if self.video_access:
            self.video_access.release()
        
        self.door_lock.cleanup()
        
        if self.tts_manager:
            self.tts_manager.cleanup()
        
        if hasattr(self.data_poster, 'session'):
            self.data_poster.session.close()
        
        self.logger.info("Enhanced security system shutdown complete")

    def cleanup(self):
        """Maintain compatibility with original cleanup method"""
        self.cleanup_enhanced()

    def add_person(self, name: str, image: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Maintain compatibility with original add_person method"""
        return self.add_person_enhanced(name, image)

    def get_system_status(self) -> Dict[str, Any]:
        """Get system status (enhanced version)"""
        return self.get_system_status_enhanced()

    def _start_continuous_recognition(self):
        """Start enhanced continuous face recognition"""
        if self.recognition_thread and self.recognition_thread.is_alive():
            self.logger.info("Enhanced continuous recognition already running")
            return
        
        self.recognition_active = True
        self.recognition_thread = Thread(
            target=self._recognition_loop,
            daemon=True,
            name="Enhanced-Face-Recognition-Thread"
        )
        self.recognition_thread.start()
        self.logger.info("Enhanced continuous face recognition started")

    def _stop_continuous_recognition(self):
        """Stop enhanced continuous face recognition"""
        self.recognition_active = False
        if self.recognition_thread and self.recognition_thread.is_alive():
            self.shutdown_event.set()
            self.recognition_thread.join(timeout=5)
            self.shutdown_event.clear()
        self.logger.info("Enhanced continuous face recognition stopped")

    def _start_server_sync(self):
        """Start enhanced background thread for server synchronization"""
        if self.sync_thread and self.sync_thread.is_alive():
            self.logger.info("Enhanced server sync already running")
            return
        
        self.sync_thread = Thread(
            target=self._sync_loop,
            daemon=True,
            name="Enhanced-Server-Sync-Thread"
        )
        self.sync_thread.start()
        self.logger.info("Enhanced server sync thread started")

    def _sync_loop(self):
        """Enhanced periodical server synchronization with DeepFace integration"""
        self.logger.info("Enhanced server sync loop started with DeepFace")
        while not self.shutdown_event.is_set():
            try:
                current_time = time.time()
                if (current_time - self.last_sync_time) >= self.sync_interval:
                    self.sync_with_server()
                    self.last_sync_time = current_time
                time.sleep(60)
            except Exception as e:
                self.logger.error(f"Enhanced server sync loop error: {e}")
                time.sleep(60)
