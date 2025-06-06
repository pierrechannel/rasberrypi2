import cv2
import numpy as np
import logging
from typing import List, Tuple

try:
    import face_recognition
    import dlib
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    print("WARNING: face_recognition not available. Falling back to mock recognition.")

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