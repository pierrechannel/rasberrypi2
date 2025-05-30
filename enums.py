from enum import Enum

class AccessResult(Enum):
    """Enumeration for access control results"""
    GRANTED = "granted"
    DENIED = "denied"
    UNKNOWN = "unknown"