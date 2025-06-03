import time
import logging
import datetime
import cv2
import base64
from typing import Any, Optional, List, Dict, Tuple
from threading import Thread, Event, Lock
from tts_manager import TTSManager
from door_lock import DoorLockController
from streaming import StreamingManager
from models import FaceRecognitionResult
from enums import AccessResult
import requests
import os
from dotenv import load_dotenv
import numpy as np
from contextlib import contextmanager
from collections import deque  # Required for recognition_history
import random  # Required for _mock_recognition

try:
    import face_recognition
    import dlib
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    print("WARNING: face_recognition not available. Falling back to mock recognition.")

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class EnhancedImageProcessor:
    """Enhanced image processing for better face recognition accuracy"""
    
    @staticmethod
    def enhance_image_quality(image: np.ndarray) -> np.ndarray:
        """Apply image enhancement techniques for better face recognition"""
        try:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            enhanced = cv2.filter2D(enhanced, -1, kernel)
            return enhanced
        except Exception as e:
            logger.error(f"Image enhancement failed: {e}")
            return image
    
    @staticmethod
    def detect_and_align_faces(image: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """Detect faces and align them for better recognition accuracy"""
        if not FACE_RECOGNITION_AVAILABLE:
            return [(image, (0, 0, image.shape[0], image.shape[1]))]
        try:
            detector = dlib.get_frontal_face_detector()
            predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)
            aligned_faces = []
            for face in faces:
                landmarks = predictor(gray, face)
                left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) 
                                   for i in range(36, 42)]).mean(axis=0).astype(int)
                right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) 
                                    for i in range(42, 48)]).mean(axis=0).astype(int)
                dy = right_eye[1] - left_eye[1]
                dx = right_eye[0] - left_eye[0]
                angle = np.degrees(np.arctan2(dy, dx))
                center = ((face.left() + face.right()) // 2, (face.top() + face.bottom()) // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                aligned = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
                face_region = aligned[face.top():face.bottom(), face.left():face.right()]
                face_location = (face.top(), face.right(), face.bottom(), face.left())
                aligned_faces.append((face_region, face_location))
            return aligned_faces if aligned_faces else [(image, (0, 0, image.shape[0], image.shape[1]))]
        except Exception as e:
            logger.error(f"Face alignment failed: {e}")
            face_locations = face_recognition.face_locations(image)
            if face_locations:
                top, right, bottom, left = face_locations[0]
                face_region = image[top:bottom, left:right]
                return [(face_region, face_locations[0])]
            return [(image, (0, 0, image.shape[0], image.shape[1]))]
    
    @staticmethod
    def optimize_face_for_encoding(face_image: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """Optimize face image for encoding generation"""
        try:
            resized = cv2.resize(face_image, target_size, interpolation=cv2.INTER_CUBIC)
            normalized = cv2.normalize(resized, None, 0, 255, cv2.NORM_MINMAX)
            gamma = 1.2
            gamma_corrected = np.power(normalized / 255.0, 1.0 / gamma)
            gamma_corrected = (gamma_corrected * 255).astype(np.uint8)
            return gamma_corrected
        except Exception as e:
            logger.error(f"Face optimization failed: {e}")
            return face_image

class RobustDataPoster:
    """Enhanced data posting with retry mechanisms - No offline storage"""
    
    def __init__(self, max_retries: int = 5, base_delay: float = 1.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.session = requests.Session()
        self.session.timeout = 15
        
    def post_with_exponential_backoff(self, url: str, data: Dict = None, files: Dict = None, 
                                    json_data: Dict = None) -> Tuple[bool, Optional[Dict]]:
        """Post data with exponential backoff retry strategy"""
        for attempt in range(self.max_retries):
            try:
                delay = self.base_delay * (2 ** attempt)
                if attempt > 0:
                    logger.info(f"Retry attempt {attempt + 1}/{self.max_retries} after {delay}s delay")
                    time.sleep(delay)
                if files:
                    response = self.session.post(url, data=data, files=files)
                elif json_data:
                    response = self.session.post(url, json=json_data, 
                                               headers={'Content-Type': 'application/json'})
                else:
                    response = self.session.post(url, data=data)
                response.raise_for_status()
                try:
                    return True, response.json()
                except:
                    return True, {"status": "success", "message": "Posted successfully"}
            except requests.exceptions.Timeout:
                logger.error(f"Timeout on attempt {attempt + 1}")
                continue
            except requests.exceptions.ConnectionError:
                logger.error(f"Connection error on attempt {attempt + 1}")
                continue
            except requests.exceptions.HTTPError as e:
                if e.response.status_code in [400, 401, 403, 404]:
                    logger.error(f"Client error {e.response.status_code}: {e.response.text}")
                    return False, {"error": f"Client error: {e.response.status_code}"}
                logger.error(f"HTTP error on attempt {attempt + 1}: {e}")
                continue
            except Exception as e:
                logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")
                continue
        return False, {"error": "Max retries exceeded"}
    
    def get_with_exponential_backoff(self, url: str, params: Dict = None) -> Tuple[bool, Optional[Dict]]:
        """GET request with exponential backoff retry strategy"""
        for attempt in range(self.max_retries):
            try:
                delay = self.base_delay * (2 ** attempt)
                if attempt > 0:
                    logger.info(f"GET retry attempt {attempt + 1}/{self.max_retries} after {delay}s delay")
                    time.sleep(delay)
                response = self.session.get(url, params=params)
                response.raise_for_status()
                try:
                    return True, response.json()
                except:
                    return True, {"status": "success", "message": "Request successful"}
            except requests.exceptions.Timeout:
                logger.error(f"GET timeout on attempt {attempt + 1}")
                continue
            except requests.exceptions.ConnectionError:
                logger.error(f"GET connection error on attempt {attempt + 1}")
                continue
            except requests.exceptions.HTTPError as e:
                if e.response.status_code in [400, 401, 403, 404]:
                    logger.error(f"GET client error {e.response.status_code}: {e.response.text}")
                    return False, {"error": f"Client error: {e.response.status_code}"}
                logger.error(f"GET HTTP error on attempt {attempt + 1}: {e}")
                continue
            except Exception as e:
                logger.error(f"GET unexpected error on attempt {attempt + 1}: {e}")
                continue
        return False, {"error": "Max retries exceeded"}

class SecuritySystem:
    """Enhanced security system with improved face recognition - No offline storage"""

    def __init__(self, device_id: str = "RPI_001"):
        self.device_id = device_id
        self.logger = logging.getLogger(f"{__name__}.{device_id}")
        self.image_processor = EnhancedImageProcessor()
        self.data_poster = RobustDataPoster()
        self.tts_manager = TTSManager()
        self.door_lock = DoorLockController()
        self.streaming_manager = StreamingManager(self)
        self.video_access = None
        self.camera_error_count = 0
        self.max_camera_errors = 5
        self._setup_enhanced_camera()
        self.known_face_names = []
        self.known_face_encodings = []
        self.known_face_ids = {}
        self.face_encoding_quality_scores = {}
        self.last_recognition_time = 0
        self.recognition_cooldown = 10
        self.recognition_history = deque(maxlen=5)
        self.confidence_threshold = 0.7
        self.face_data_lock = Lock()
        self.recognition_active = True
        self.recognition_thread = None
        self.shutdown_event = Event()
        self.server_users_url = os.getenv("SERVER_USERS_URL", "https://apps.mediabox.bi:26875/administration/warehouse_users")
        self.access_log_url = os.getenv("SERVER_ACCESS_LOG_URL", "https://apps.mediabox.bi:26875/warehouse_acces/create")
        self.sync_interval = 180
        self.sync_thread = None
        self.last_sync_time = 0
        self.synced_person_ids = set()
        if self.tts_manager:
            self.tts_manager.speak("system_startup")
        self.logger.info("Enhanced security system initialized")
        if FACE_RECOGNITION_AVAILABLE:
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
                    self.video_access.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.video_access.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    self.video_access.set(cv2.CAP_PROP_FPS, 30)
                    self.video_access.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)
                    self.video_access.set(cv2.CAP_PROP_CONTRAST, 0.5)
                    self.video_access.set(cv2.CAP_PROP_SATURATION, 0.5)
                    self.video_access.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
                    self.video_access.set(cv2.CAP_PROP_AUTOFOCUS, 1)
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
        """Load known faces with enhanced processing"""
        self.logger.info("Loading known face encodings with enhanced processing")
        # Implementation would load from server via sync_with_server
        self.sync_with_server()

    def process_face_recognition_enhanced(self, frame: np.ndarray) -> List[FaceRecognitionResult]:
        """Enhanced face recognition with improved accuracy"""
        results = []
        if frame is None:
            return results
        if not FACE_RECOGNITION_AVAILABLE:
            return self._mock_recognition(frame)
        try:
            enhanced_frame = self.image_processor.enhance_image_quality(frame)
            aligned_faces = self.image_processor.detect_and_align_faces(enhanced_frame)
            with self.face_data_lock:
                known_encodings = [enc for enc in self.known_face_encodings if enc is not None]
                known_names = self.known_face_names.copy()
            for face_image, face_location in aligned_faces:
                optimized_face = self.image_processor.optimize_face_for_encoding(face_image)
                rgb_face = cv2.cvtColor(optimized_face, cv2.COLOR_BGR2RGB)
                face_encodings = face_recognition.face_encodings(
                    rgb_face, 
                    num_jitters=10, 
                    model='large'
                )
                if not face_encodings:
                    continue
                face_encoding = face_encodings[0]
                name, confidence, access_result = self._enhanced_face_matching(
                    face_encoding, known_encodings, known_names
                )
                result = FaceRecognitionResult(
                    person_id=self.known_face_ids.get(name, f"ID_{hash(name) % 1000}"),
                    name=name,
                    confidence=confidence,
                    location=face_location,
                    access_result=access_result
                )
                results.append(result)
                self.logger.info(f"Enhanced recognition: {name} with confidence {confidence:.2f}")
        except Exception as e:
            self.logger.error(f"Enhanced face recognition error: {e}")
        return results

    def _enhanced_face_matching(self, face_encoding: np.ndarray, known_encodings: List, 
                              known_names: List[str]) -> Tuple[str, float, AccessResult]:
        """Enhanced face matching with multiple validation techniques"""
        if not known_encodings:
            return "Unknown", 0.0, AccessResult.UNKNOWN
        try:
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            tolerance_strict = 0.4
            tolerance_normal = 0.6
            best_match_index = np.argmin(face_distances)
            best_distance = face_distances[best_match_index]
            confidence = (1 - best_distance) * 100
            if best_distance <= tolerance_strict:
                name = known_names[best_match_index]
                return name, confidence, AccessResult.GRANTED
            elif best_distance <= tolerance_normal:
                name = known_names[best_match_index]
                if self._validate_with_history(name, confidence):
                    return name, confidence, AccessResult.GRANTED
                else:
                    return name, confidence, AccessResult.UNKNOWN
            else:
                return "Unknown", confidence, AccessResult.UNKNOWN
        except Exception as e:
            self.logger.error(f"Enhanced matching error: {e}")
            return "Unknown", 0.0, AccessResult.UNKNOWN

    def _validate_with_history(self, name: str, confidence: float) -> bool:
        """Validate recognition using historical data"""
        self.recognition_history.append({"name": name, "confidence": confidence, "timestamp": time.time()})
        recent_recognitions = [r for r in self.recognition_history 
                             if time.time() - r["timestamp"] < 30]
        same_person_count = sum(1 for r in recent_recognitions if r["name"] == name)
        avg_confidence = np.mean([r["confidence"] for r in recent_recognitions if r["name"] == name]) if recent_recognitions else 0.0
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
        """Enhanced access attempt processing"""
        try:
            with self.recognition_cooldown_check():
                results = self.process_face_recognition_enhanced(frame)
                response_data = {
                    "faces_detected": len(results),
                    "results": [],
                    "timestamp": datetime.datetime.now().isoformat(),
                    "device_id": self.device_id,
                    "processing_enhanced": True
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
                        "enhanced_processing": True
                    })
                return response_data
        except Exception as e:
            self.logger.error(f"Enhanced access processing error: {e}")
            raise

    def sync_with_server(self):
        """Enhanced server synchronization - GET users only"""
        self.logger.info("Starting enhanced server sync - fetching users")
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
                self.logger.info(f"Enhanced sync: Added {new_count} new persons")
                if self.tts_manager:
                    self.tts_manager.speak_custom(f"Enhanced sync complete: {new_count} new persons")
        except Exception as e:
            self.logger.error(f"Enhanced server sync processing error: {e}")

    def _process_server_persons(self, persons: List[Dict]) -> int:
        """Process persons from server with enhanced validation"""
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
                encoding = self._generate_enhanced_encoding(image)
                if encoding is None and FACE_RECOGNITION_AVAILABLE:
                    continue
                with self.face_data_lock:
                    self.known_face_names.append(name)
                    self.known_face_encodings.append(encoding)
                    self.known_face_ids[name] = person_id
                    self.synced_person_ids.add(person_id)
                new_count += 1
                self.logger.info(f"Enhanced processing: Added {name}")
            except Exception as e:
                self.logger.error(f"Error processing person: {e}")
                continue
        return new_count

    def _download_and_process_image(self, photo_url: str) -> Optional[np.ndarray]:
        """Download and enhance image for better encoding"""
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

    def _generate_enhanced_encoding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Generate high-quality face encoding"""
        if not FACE_RECOGNITION_AVAILABLE:
            return None
        try:
            aligned_faces = self.image_processor.detect_and_align_faces(image)
            for face_image, _ in aligned_faces:
                optimized_face = self.image_processor.optimize_face_for_encoding(face_image)
                rgb_face = cv2.cvtColor(optimized_face, cv2.COLOR_BGR2RGB)
                encodings = face_recognition.face_encodings(
                    rgb_face,
                    num_jitters=15,
                    model='large'
                )
                if encodings:
                    return encodings[0]
            return None
        except Exception as e:
            self.logger.error(f"Enhanced encoding generation error: {e}")
            return None

    def _mock_recognition(self, frame: np.ndarray) -> List[FaceRecognitionResult]:
        """Enhanced mock recognition for testing"""
        results = []
        height, width = frame.shape[:2]
        face_location = (
            random.randint(50, height//3),
            random.randint(2*width//3, width-50),
            random.randint(2*height//3, height-50),
            random.randint(50, width//3)
        )
        with self.face_data_lock:
            names = self.known_face_names + ["Unknown"]
        name = random.choice(names) if names else "Unknown"
        if name != "Unknown" and name in self.known_face_names:
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
        """Enhanced continuous face recognition loop"""
        self.logger.info("Enhanced face recognition loop started")
        frame_skip_counter = 0
        process_every_n_frames = 3  # Process every 3rd frame for better performance
        while self.recognition_active and not self.shutdown_event.is_set():
            try:
                frame = self.get_frame()
                if frame is not None:
                    frame_skip_counter += 1
                    if frame_skip_counter >= process_every_n_frames:
                        frame_skip_counter = 0
                        self.process_access_attempt(frame)
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
                self.logger.error(f"Enhanced recognition loop error: {e}")
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
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
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
        recent_activity = len([r for r in self.recognition_history 
                              if current_time - r.get("timestamp", 0) < 60])
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
        """Enhanced person addition with quality validation, no offline storage"""
        try:
            if not name or not isinstance(name, str):
                raise ValueError("Valid name is required")
            name = name.strip()
            if len(name) < 2:
                raise ValueError("Name must be at least 2 characters")
            with self.face_data_lock:
                if name in self.known_face_names:
                    raise ValueError(f"{name} already exists")
                face_encoding = None
                quality_score = 0.0
                if image is not None:
                    enhanced_image = self.image_processor.enhance_image_quality(image)
                    quality_score = self._calculate_frame_quality(enhanced_image)
                    if quality_score < 0.3:
                        raise ValueError("Image quality too low for reliable recognition")
                    if FACE_RECOGNITION_AVAILABLE:
                        face_encoding = self._generate_enhanced_encoding(enhanced_image)
                        if face_encoding is None:
                            raise ValueError("No face detected or encoding failed")
                    self.logger.info(f"Person {name} added with quality score: {quality_score:.2f}")
                self.known_face_names.append(name)
                self.known_face_encodings.append(face_encoding)
                self.face_encoding_quality_scores[name] = quality_score
                person_id = f"LOCAL_{hash(name + str(time.time())) % 10000}"
                self.known_face_ids[name] = person_id
            log_entry = {
                "person_id": person_id,
                "name": name,
                "timestamp": datetime.datetime.now().isoformat(),
                "action": "person_added_enhanced",
                "has_encoding": face_encoding is not None,
                "quality_score": quality_score,
                "metadata": metadata or {}
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
                "known_persons_count": len(self.known_face_names),
                "has_encoding": face_encoding is not None,
                "enhanced_processing": True
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
            known_count = len(self.known_face_names)
            avg_quality = np.mean(list(self.face_encoding_quality_scores.values())) if self.face_encoding_quality_scores else 0.0
        return {
            "device_id": self.device_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "camera_status": "active" if (self.video_access and self.video_access.isOpened()) else "error",
            "recognition_active": self.recognition_active,
            "known_persons": known_count,
            "average_encoding_quality": round(avg_quality, 2),
            "synced_persons": len(self.synced_person_ids),
            "recognition_history_size": len(self.recognition_history),
            "last_sync_time": datetime.datetime.fromtimestamp(self.last_sync_time).isoformat() if self.last_sync_time else None,
            "enhancement_features": {
                "image_enhancement": True,
                "face_alignment": FACE_RECOGNITION_AVAILABLE,
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
            self.shutdown_event.clear() # Clear event for future use
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
        """Enhanced periodical server synchronization"""
        self.logger.info("Enhanced server sync loop started")
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