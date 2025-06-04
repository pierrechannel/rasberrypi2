import signal
import logging
from security_system import SecuritySystem
from api import create_app
from streaming import StreamingManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize security system
security_system = SecuritySystem()
streaming-manager=StreamingManager()

# Create Flask app
app = create_app(security_system)

# In your API module
@app.route('/streaming/start', methods=['POST'])
def start_streaming():
    if streaming_manager.start_streaming():
        return {"status": "success", "message": "Streaming started"}
    return {"status": "error", "message": "Failed to start streaming"}, 400

@app.route('/streaming/stop', methods=['POST'])
def stop_streaming():
    if streaming_manager.stop_streaming():
        return {"status": "success", "message": "Streaming stopped"}
    return {"status": "error", "message": "Streaming not active"}, 400

@app.route('/streaming/stats', methods=['GET'])
def get_streaming_stats():
    return streaming_manager.get_streaming_stats()

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
            debug=False,  # Set to False for production
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


