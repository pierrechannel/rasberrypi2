from dataclasses import dataclass
from typing import Tuple
from enums import AccessResult

@dataclass
class FaceRecognitionResult:
    """Data class for face recognition results"""
    person_id: str
    name: str
    confidence: float
    location: Tuple[int, int, int, int]  # (top, right, bottom, left)
    access_result: AccessResult