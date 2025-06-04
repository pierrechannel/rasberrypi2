# api.py (partial, for reference)
from flask import Flask, Response, jsonify, request
import cv2
import numpy as np
from security_system import SecuritySystem
from streaming import StreamingManager
import logging

logger = logging.getLogger(__name__)

def create_app(security_system: SecuritySystem) -> Flask:
    app = Flask(__name__)
    streaming_manager = StreamingManager(security_system)

    @app.route('/process_access', methods=['POST'])
    def process_access():
        try:
            if 'image' in request.files:
                file = request.files['image']
                image_data = file.read()
                nparr = np.frombuffer(image_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is None:
                    return jsonify({"success": False, "message": "Invalid image data"}), 400
            else:
                frame = security_system.get_frame()
                if frame is None:
                    return jsonify({"success": False, "message": "Failed to capture frame from camera"}), 500

            response_data = security_system.process_access_attempt(frame)

            for result in response_data.get("results", []):
                if result["access_granted"] is False and result["name"] == "Unknown":
                    logger.info(f"Logging access attempt for unknown person with confidence {result['confidence']}")

            return jsonify({"success": True, "data": response_data}), 200

        except Exception as e:
            logger.error(f"Error processing access attempt: {e}")
            return jsonify({"success": False, "message": f"Error processing access: {str(e)}"}), 500

    @app.route('/stream', methods=['GET'])
    def stream():
        def generate():
            while True:
                try:
                    frame_bytes = streaming_manager.generate_frame_with_detection()
                    if frame_bytes is None:
                        continue
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                except Exception as e:
                    logger.error(f"Streaming error: {e}")
                    break

        if not streaming_manager.active:
            streaming_manager.start_streaming()

        return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

    @app.route('/configure_stream', methods=['POST'])
    def configure_stream():
        """Configure streaming servers"""
        try:
            data = request.get_json()
            if not data or 'server_urls' not in data:
                return jsonify({
                    "success": False,
                    "message": "Missing 'server_urls' in request body"
                }), 400

            config_updates = {}
            if 'server_urls' in data:
                config_updates['server_urls'] = data['server_urls']
            if 'fps' in data:
                config_updates['fps'] = data['fps']
            if 'quality' in data:
                config_updates['quality'] = data['quality']
            if 'detection_enabled' in data:
                config_updates['detection_enabled'] = data['detection_enabled']
            if 'timeout' in data:
                config_updates['timeout'] = data['timeout']

            streaming_manager.set_config(**config_updates)
            streaming_manager.stop_streaming()
            streaming_manager.start_streaming()

            return jsonify({
                "success": True,
                "message": "Streaming configuration updated",
                "config": {
                    "server_urls": streaming_manager.config.server_urls,
                    "fps": streaming_manager.config.fps,
                    "quality": streaming_manager.config.quality,
                    "detection_enabled": streaming_manager.config.detection_enabled,
                    "timeout": streaming_manager.config.timeout
                }
            }), 200

        except Exception as e:
            logger.error(f"Error configuring stream: {e}")
            return jsonify({
                "success": False,
                "message": f"Error configuring stream: {str(e)}"
            }), 500

    @app.route('/status', methods=['GET'])
    def status():
        """Get system and streaming status"""
        try:
            system_status = security_system.get_system_status()
            streaming_status = streaming_manager.get_streaming_stats()
            return jsonify({
                "success": True,
                "system_status": system_status,
                "streaming_status": streaming_status
            }), 200
        except Exception as e:
            logger.error(f"Error retrieving status: {e}")
            return jsonify({
                "success": False,
                "message": f"Error retrieving status: {str(e)}"
            }), 500

    @app.route('/shutdown', methods=['POST'])
    def shutdown():
        """Shut down the security system and stop streaming"""
        try:
            streaming_manager.stop_streaming()
            security_system.cleanup()
            return jsonify({
                "success": True,
                "message": "System shutdown successfully"
            }), 200
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            return jsonify({
                "success": False,
                "message": f"Error during shutdown: {str(e)}"
            }), 500

    return app