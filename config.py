from dataclasses import dataclass
from typing import Optional

@dataclass
class StreamConfig:
    """Configuration for video streaming"""
    fps: int = 10
    quality: int = 80
    detection_enabled: bool = True
    server_url: Optional[str] = None