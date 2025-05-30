import time
import threading
import logging
from threading import Timer, Event
from collections import deque

try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("WARNING: pyttsx3 not available. TTS will be simulated.")

logger = logging.getLogger(__name__)

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