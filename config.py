from dataclasses import dataclass
from typing import Optional

@dataclass
class StreamConfig:
    def __init__(self):
        self.server_urls = [
            "http://existing-server:8080",  # Existing server
            "http://additional-server:8081"  # New server
        ]
        self.fps = 15
        self.quality = 90
        self.detection_enabled = True
        self.timeout = 5