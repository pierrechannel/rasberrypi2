from flask import Flask, request, jsonify, Response
import datetime
import base64
from security_system import SecuritySystem
from tts_manager import TTS_AVAILABLE
from door_lock import GPIO_AVAILABLE

def create_app(security_system: SecuritySystem):
    """Create and configure Flask application"""
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

    @app.route('/health', methods=['GET'])
    def health_check():
        """System health check endpoint"""
        return jsonify({
            "status": "healthy",
            "device_id": security_system.device_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "known_persons": len(security_system.known_face_names),
            "camera_status": "active" if security_system.video_capture 
                            and security_system.video_capture.isOpened() else "inactive",
            "tts_active": security_system.tts_manager.is_active,
            "backend_online": security_system.backend_online,
            "offline_logs_count": len(security_system.offline_logs),
            "streaming_active": security_system.streaming_manager.active,
            "door_status": "open" if security_system.door_lock.is_door_open else "locked"
        })

    @app.route('/capture', methods=['GET'])
    def capture_and_recognize():
        """Capture image and perform face recognition"""
        try:
            frame = security_system.get_frame()
            if frame is None:
                return jsonify({"error": "Unable to capture image"}), 500
            
            result = security_system.process_access_attempt(frame)
            return jsonify(result)
            
        except Exception as e:
            if "cooldown" in str(e).lower():
                return jsonify({"error": str(e)}), 429
            
            security_system.logger.error(f"Capture error: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/stream/config', methods=['GET', 'POST'])
    def stream_config():
        """Get or update streaming configuration"""
        if request.method == 'GET':
            return jsonify({
                "fps": security_system.streaming_manager.config.fps,
                "quality": security_system.streaming_manager.config.quality,
                "detection_enabled": security_system.streaming_manager.config.detection_enabled,
                "server_url": security_system.streaming_manager.config.server_url,
                "active": security_system.streaming_manager.active
            })
        
        elif request.method == 'POST':
            try:
                data = request.get_json() or {}
                
                # Update configuration
                config_updates = {}
                for key in ['fps', 'quality', 'detection_enabled', 'server_url']:
                    if key in data:
                        if key == 'fps' and data[key]:
                            config_updates[key] = max(1, min(30, int(data[key])))
                        elif key == 'quality' and data[key]:
                            config_updates[key] = max(10, min(100, int(data[key])))
                        else:
                            config_updates[key] = data[key]
                
                if config_updates:
                    security_system.streaming_manager.set_config(**config_updates)
                
                return jsonify({
                    "message": "Configuration updated",
                    "config": {
                        "fps": security_system.streaming_manager.config.fps,
                        "quality": security_system.streaming_manager.config.quality,
                        "detection_enabled": security_system.streaming_manager.config.detection_enabled,
                        "server_url": security_system.streaming_manager.config.server_url
                    }
                })
            except Exception as e:
                return jsonify({"error": str(e)}), 500

    @app.route('/stream/live')
    def live_stream():
        """Live video stream endpoint"""
        def generate():
            # Ensure streaming is active for local viewing
            if not security_system.streaming_manager.active:
                security_system.streaming_manager.active = True
            
            try:
                while True:
                    frame_bytes = security_system.streaming_manager.generate_frame_with_detection()
                    if frame_bytes:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    time.sleep(1.0 / security_system.streaming_manager.config.fps)
            except GeneratorExit:
                pass
            except Exception as e:
                security_system.logger.error(f"Live stream error: {e}")
        
        return Response(
            generate(),
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )

    @app.route('/stream/snapshot', methods=['GET'])
    def get_snapshot():
        """Get current frame snapshot"""
        try:
            frame_bytes = security_system.streaming_manager.generate_frame_with_detection()
            if frame_bytes is None:
                return jsonify({"error": "Unable to capture image"}), 500
            
            frame_b64 = base64.b64encode(frame_bytes).decode('utf-8')
            return jsonify({
                "image": frame_b64,
                "timestamp": datetime.datetime.now().isoformat(),
                "device_id": security_system.device_id,
                "detection_enabled": security_system.streaming_manager.config.detection_enabled
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/status', methods=['GET'])
    def get_system_status():
        """Get comprehensive system status"""
        return jsonify({
            "device_id": security_system.device_id,
            "system_time": datetime.datetime.now().isoformat(),
            "known_persons": len(security_system.known_face_names),
            "camera_active": security_system.video_capture and security_system.video_capture.isOpened(),
            "tts_active": security_system.tts_manager.is_active,
            "backend_online": security_system.backend_online,
            "offline_logs_pending": len(security_system.offline_logs),
            "last_recognition": security_system.last_recognition_time,
            "backend_url": security_system.backend_api_url,
            "recognition_cooldown": security_system.recognition_cooldown,
            "door": {
                "is_open": security_system.door_lock.is_door_open,
                "gpio_available": GPIO_AVAILABLE
            },
            "streaming": {
                "active": security_system.streaming_manager.active,
                "fps": security_system.streaming_manager.config.fps,
                "quality": security_system.streaming_manager.config.quality,
                "detection_enabled": security_system.streaming_manager.config.detection_enabled,
                "server_url": security_system.streaming_manager.config.server_url,
                "error_count": security_system.streaming_manager.error_count
            },
            "tts": {
                "active": security_system.tts_manager.is_active,
                "queue_size": len(security_system.tts_manager.tts_queue),
                "available_messages": list(security_system.tts_manager.VOICE_MESSAGES.keys())
            }
        })

    @app.route('/sync', methods=['POST'])
    def manual_sync():
        """Manually sync with backend"""
        try:
            success = security_system.sync_with_backend()
            return jsonify({
                "success": success,
                "message": "Synchronization completed" if success else "Synchronization failed",
                "known_persons": len(security_system.known_face_names),
                "offline_logs_processed": 0 if success else len(security_system.offline_logs)
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/logs/offline', methods=['GET'])
    def get_offline_logs():
        """Get offline access logs"""
        return jsonify({
            "offline_logs": security_system.offline_logs,
            "count": len(security_system.offline_logs),
            "device_id": security_system.device_id
        })

    @app.route('/logs/clear', methods=['POST'])
    def clear_offline_logs():
        """Clear offline logs"""
        try:
            count = len(security_system.offline_logs)
            security_system.offline_logs.clear()
            return jsonify({
                "message": f"Cleared {count} offline logs",
                "remaining_logs": len(security_system.offline_logs)
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/tts/queue/clear', methods=['POST'])
    def clear_tts_queue():
        """Clear TTS message queue"""
        try:
            security_system.tts_manager.clear_queue()
            return jsonify({
                "message": "TTS queue cleared",
                "success": True
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/tts/messages', methods=['GET'])
    def get_tts_messages():
        """Get available TTS messages"""
        return jsonify({
            "predefined_messages": security_system.tts_manager.VOICE_MESSAGES,
            "queue_size": len(security_system.tts_manager.tts_queue),
            "tts_active": security_system.tts_manager.is_active
        })

    @app.route('/door/lock', methods=['POST'])
    def manual_lock():
        """Manually lock door"""
        try:
            success = security_system.door_lock.lock_door()
            return jsonify({
                "success": success,
                "message": "Door locked" if success else "Failed to lock door",
                "door_status": "locked" if success else "unknown"
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/door/unlock', methods=['POST'])
    def manual_unlock():
        """Manually unlock door"""
        try:
            data = request.get_json() or {}
            duration = data.get('duration', 5)
            name = data.get('name', 'Manual Override')
            
            success = security_system.door_lock.unlock_door(duration=duration, name=name)
            return jsonify({
                "success": success,
                "message": f"Door unlocked for {duration} seconds" if success else "Failed to unlock door",
                "duration": duration,
                "door_status": "unlocked" if success else "unknown"
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/door/emergency_unlock', methods=['POST'])
    def emergency_unlock():
        """Emergency door unlock endpoint"""
        try:
            success = security_system.door_lock.emergency_unlock()
            return jsonify({
                "success": success,
                "message": "Emergency unlock activated" if success else "Emergency unlock failed"
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/announce', methods=['POST'])
    def make_announcement():
        """Make TTS announcement"""
        try:
            data = request.get_json()
            if not data or 'message' not in data:
                return jsonify({"error": "Message required"}), 400
            
            message = data['message']
            message_type = data.get('type', 'custom')
            priority = data.get('priority', False)
            
            if message_type == 'predefined':
                success = security_system.tts_manager.speak(message, priority=priority)
            else:
                success = security_system.tts_manager.speak_custom(message, priority=priority)
            
            return jsonify({
                "message": "Announcement queued" if success else "TTS not available",
                "success": success
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/stream/start', methods=['POST'])
    def start_streaming():
        """Start video streaming"""
        try:
            data = request.get_json() or {}
            
            # Update streaming configuration
            config_updates = {}
            for key in ['fps', 'quality', 'detection_enabled', 'server_url']:
                if key in data:
                    config_updates[key] = data[key]
            
            if config_updates:
                security_system.streaming_manager.set_config(**config_updates)
            
            success = security_system.streaming_manager.start_streaming()
            
            if success and security_system.tts_manager:
                security_system.tts_manager.speak("streaming_started")
            
            return jsonify({
                "success": success,
                "message": "Streaming started" if success else "Streaming already active",
                "config": {
                    "fps": security_system.streaming_manager.config.fps,
                    "quality": security_system.streaming_manager.config.quality,
                    "detection_enabled": security_system.streaming_manager.config.detection_enabled,
                    "server_url": security_system.streaming_manager.config.server_url
                }
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/stream/stop', methods=['POST'])
    def stop_streaming():
        """Stop video streaming"""
        try:
            success = security_system.streaming_manager.stop_streaming()
            
            if success and security_system.tts_manager:
                security_system.tts_manager.speak("streaming_stopped")
            
            return jsonify({
                "success": success,
                "message": "Streaming stopped",
                "timestamp": datetime.datetime.now().isoformat()
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/system/restart', methods=['POST'])
    def restart_system():
        """Restart system components"""
        try:
            data = request.get_json() or {}
            component = data.get('component', 'all')
            
            results = {}
            
            if component in ['all', 'camera']:
                try:
                    if security_system.video_capture:
                        security_system.video_capture.release()
                    security_system._setup_camera()
                    results['camera'] = 'restarted'
                except Exception as e:
                    results['camera'] = f'error: {str(e)}'
            
            if component in ['all', 'tts']:
                try:
                    security_system.tts_manager.cleanup()
                    security_system.tts_manager = TTSManager()
                    security_system.door_lock.tts_manager = security_system.tts_manager
                    results['tts'] = 'restarted'
                except Exception as e:
                    results['tts'] = f'error: {str(e)}'
            
            if component in ['all', 'streaming']:
                try:
                    security_system.streaming_manager.stop_streaming()
                    security_system.streaming_manager = StreamingManager(security_system)
                    results['streaming'] = 'restarted'
                except Exception as e:
                    results['streaming'] = f'error: {str(e)}'
            
            return jsonify({
                "message": f"System component(s) restart attempted",
                "component": component,
                "results": results
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    

    @app.route('/config', methods=['GET', 'POST'])
    def system_config():
        """Get or update system configuration"""
        if request.method == 'GET':
            return jsonify({
                "device_id": security_system.device_id,
                "recognition_cooldown": security_system.recognition_cooldown,
                "known_persons": security_system.known_face_names,
                "backend_url": security_system.backend_api_url,
                "gpio_available": GPIO_AVAILABLE,
                "tts_available": TTS_AVAILABLE
            })
        
        elif request.method == 'POST':
            try:
                data = request.get_json() or {}
                updated = []
                
                if 'recognition_cooldown' in data:
                    security_system.recognition_cooldown = max(1, int(data['recognition_cooldown']))
                    updated.append('recognition_cooldown')
                
                if 'backend_url' in data:
                    security_system.backend_api_url = data['backend_url']
                    updated.append('backend_url')
                
                return jsonify({
                    "message": f"Updated configuration: {', '.join(updated)}",
                    "updated_fields": updated
                })
            except Exception as e:
                return jsonify({"error": str(e)}), 500
        # Add to api.py in the create_app function before return app
    @app.route('/recognition/control', methods=['POST'])
    def recognition_control():
        """Control continuous face recognition"""
        try:
            data = request.get_json() or {}
            action = data.get('action')  # 'start' or 'stop'
            
            if action == 'start':
                security_system._start_continuous_recognition()
                return jsonify({
                    "success": True,
                    "message": "Continuous face recognition started",
                    "active": security_system.recognition_active
                })
            elif action == 'stop':
                security_system._stop_continuous_recognition()
                return jsonify({
                    "success": True,
                    "message": "Continuous face recognition stopped",
                    "active": security_system.recognition_active
                })
            else:
                return jsonify({"error": "Invalid action. Use 'start' or 'stop'"}), 400
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/recognition/status', methods=['GET'])
    def recognition_status():
        """Get continuous face recognition status"""
        return jsonify({
            "active": security_system.recognition_active,
            "last_recognition_time": security_system.last_recognition_time,
            "known_persons_count": len(security_system.known_face_names)
        })

    @app.route('/camera/status', methods=['GET'])
    def camera_status():
        return jsonify({
            "camera_active": security_system.video_capture and security_system.video_capture.isOpened(),
            "error_count": security_system.camera_error_count,
            "last_error": "None" if security_system.camera_error_count == 0 else "Camera unavailable"
        })
    @app.route('/sync/status', methods=['GET'])
    def sync_status():
        return jsonify({
            "last_sync_time": security_system.last_sync_time,
            "synced_persons": len(security_system.known_face_names),
            "sync_error": "None" if security_system.last_sync_time else "Sync failed"
        })

    return app