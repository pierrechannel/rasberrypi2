import threading
import logging
from threading import Timer
from tts_manager import TTSManager

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    print("WARNING: RPi.GPIO not available. Door lock control will be simulated.")

logger = logging.getLogger(__name__)

class DoorLockController:
    """Enhanced door lock controller with TTS integration"""
    
    def __init__(self, relay_pin: int = 18, led_green_pin: int = 16, 
                 led_red_pin: int = 20, buzzer_pin: int = 21, 
                 tts_manager: TTSManager = None):
        self.relay_pin = relay_pin
        self.led_green_pin = led_green_pin
        self.led_red_pin = led_red_pin
        self.buzzer_pin = buzzer_pin
        self.lock_timer = None
        self.is_door_open = False
        self.tts_manager = tts_manager
        self.lock = threading.Lock()
        
        if GPIO_AVAILABLE:
            self._setup_gpio()
        else:
            logger.info("GPIO simulation mode activated")

    def _setup_gpio(self):
        """Initialize GPIO pins"""
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            
            # Setup output pins
            for pin in [self.relay_pin, self.led_green_pin, self.led_red_pin, self.buzzer_pin]:
                GPIO.setup(pin, GPIO.OUT)
                GPIO.output(pin, GPIO.LOW)
            
            # Start in locked state
            self._set_locked_state()
            logger.info("GPIO pins initialized successfully")
            
        except Exception as e:
            logger.error(f"GPIO setup failed: {e}")

    def unlock_door(self, duration: int = 5, name: str = None) -> bool:
        """Unlock door with TTS announcement"""
        with self.lock:
            try:
                if GPIO_AVAILABLE:
                    GPIO.output(self.relay_pin, GPIO.HIGH)
                    GPIO.output(self.led_green_pin, GPIO.HIGH)
                    GPIO.output(self.led_red_pin, GPIO.LOW)
                
                self.is_door_open = True
                logger.info(f"Door unlocked for {duration} seconds")
                
                # TTS announcements
                if self.tts_manager:
                    if name and name != "Unknown":
                        self.tts_manager.speak("welcome", name=name)
                    self.tts_manager.speak("door_unlocked", duration=duration)
                
                # Cancel existing timer
                if self.lock_timer:
                    self.lock_timer.cancel()
                
                # Set new lock timer
                self.lock_timer = Timer(duration, self.lock_door)
                self.lock_timer.start()
                
                return True
                
            except Exception as e:
                logger.error(f"Failed to unlock door: {e}")
                return False

    def lock_door(self) -> bool:
        """Lock door with TTS announcement"""
        with self.lock:
            try:
                self._set_locked_state()
                self.is_door_open = False
                logger.info("Door locked")
                
                if self.tts_manager:
                    self.tts_manager.speak("door_locked")
                
                return True
                
            except Exception as e:
                logger.error(f"Failed to lock door: {e}")
                return False

    def _set_locked_state(self):
        """Set GPIO pins to locked state"""
        if GPIO_AVAILABLE:
            GPIO.output(self.relay_pin, GPIO.LOW)
            GPIO.output(self.led_green_pin, GPIO.LOW)
            GPIO.output(self.led_red_pin, GPIO.HIGH)

    def handle_access_denied(self, reason: str = "access_denied"):
        """Handle access denied with TTS instead of beep"""
        logger.warning(f"Access denied: {reason}")
        if self.tts_manager:
            self.tts_manager.speak(reason, priority=True)

    def handle_unknown_person(self):
        """Handle unknown person detection with TTS"""
        logger.info("Unknown person detected")
        if self.tts_manager:
            self.tts_manager.speak("unknown_person", priority=True)

    def emergency_unlock(self) -> bool:
        """Emergency unlock without timer"""
        with self.lock:
            if self.lock_timer:
                self.lock_timer.cancel()
            
            if GPIO_AVAILABLE:
                GPIO.output(self.relay_pin, GPIO.HIGH)
                GPIO.output(self.led_green_pin, GPIO.HIGH)
                GPIO.output(self.led_red_pin, GPIO.LOW)
            
            self.is_door_open = True
            logger.warning("Emergency door unlock activated")
            
            if self.tts_manager:
                self.tts_manager.speak_custom("Emergency unlock activated", priority=True)
            
            return True

    def cleanup(self):
        """Cleanup door lock resources"""
        logger.info("Cleaning up door lock controller")
        
        if self.lock_timer:
            self.lock_timer.cancel()
        
        if GPIO_AVAILABLE:
            self.lock_door()
            GPIO.cleanup()