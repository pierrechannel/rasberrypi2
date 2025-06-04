import signal
import logging
from security_system import SecuritySystem
from api import create_app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize security system
security_system = SecuritySystem()

# Create Flask app
app = create_app(security_system)

def signal_handler(sig, frame):
    """Handle shutdown signals"""
    logger.info("Shutting down server...")
    security_system.cleanup()
    exit(0)

if __name__ == '__main__':
    # Register shutdown handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        logger.info("Starting security system server...")
        app.run(
            host='0.0.0.0',
            port=5002,
            debug=False,
            threaded=True,
            use_reloader=False
        )
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
    finally:
        logger.info("Shutting down server...")
        security_system.cleanup()