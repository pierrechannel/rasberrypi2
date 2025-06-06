# Raspberry Pi Camera Test with OpenCV
# Install required packages first:
# sudo apt update
# sudo apt install python3-opencv python3-pip
# pip3 install opencv-python

import cv2
import numpy as np
import time
import os

def test_camera_basic():q
    """Basic camera test - display live feed"""
    print("Testing basic camera functionality...")
    
    # Try different camera indices (0, 1, 2) if one doesn't work
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return False
    
    # Set camera properties (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("Camera opened successfully!")
    print("Press 'q' to quit, 's' to save image, 'i' for info")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        # Display frame info on image
        height, width = frame.shape[:2]
        info_text = f"Resolution: {width}x{height}"
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (0, 255, 0), 2)
        
        cv2.imshow('Raspberry Pi Camera Test', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save current frame
            filename = f"rpi_capture_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Image saved as {filename}")
        elif key == ord('i'):
            print_camera_info(cap)
    
    cap.release()
    cv2.destroyAllWindows()
    return True

def print_camera_info(cap):
    """Print detailed camera information"""
    print("\n=== Camera Information ===")
    print(f"Frame Width: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
    print(f"Frame Height: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    print(f"FPS: {cap.get(cv2.CAP_PROP_FPS)}")
    print(f"Brightness: {cap.get(cv2.CAP_PROP_BRIGHTNESS)}")
    print(f"Contrast: {cap.get(cv2.CAP_PROP_CONTRAST)}")
    print(f"Saturation: {cap.get(cv2.CAP_PROP_SATURATION)}")
    print(f"Exposure: {cap.get(cv2.CAP_PROP_EXPOSURE)}")
    print("========================\n")

def test_camera_with_effects():
    """Test camera with various OpenCV effects"""
    print("Testing camera with effects...")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return False
    
    effect_mode = 0
    effects = ["Normal", "Grayscale", "Blur", "Edge Detection", "HSV"]
    
    print("Press 'e' to cycle effects, 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Apply different effects based on mode
        if effect_mode == 0:  # Normal
            display_frame = frame.copy()
        elif effect_mode == 1:  # Grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            display_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        elif effect_mode == 2:  # Blur
            display_frame = cv2.GaussianBlur(frame, (15, 15), 0)
        elif effect_mode == 3:  # Edge Detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            display_frame = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        elif effect_mode == 4:  # HSV
            display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Add effect name to image
        effect_text = f"Effect: {effects[effect_mode]}"
        cv2.putText(display_frame, effect_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Camera Effects Test', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('e'):
            effect_mode = (effect_mode + 1) % len(effects)
            print(f"Switched to: {effects[effect_mode]}")
    
    cap.release()
    cv2.destroyAllWindows()
    return True

def test_motion_detection():
    """Simple motion detection test"""
    print("Testing motion detection...")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return False
    
    # Initialize background subtractor
    back_sub = cv2.createBackgroundSubtractorMOG2()
    
    print("Move in front of camera to test motion detection")
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Apply background subtraction
        fg_mask = back_sub.apply(frame)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, 
                                     cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw bounding boxes around detected motion
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Filter small movements
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "Motion Detected", (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Show both original and mask
        cv2.imshow('Motion Detection', frame)
        cv2.imshow('Motion Mask', fg_mask)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return True

def capture_images(count=5, interval=2):
    """Capture multiple images with time interval"""
    print(f"Capturing {count} images with {interval} second intervals...")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return False
    
    # Create directory for captures
    capture_dir = "rpi_captures"
    os.makedirs(capture_dir, exist_ok=True)
    
    for i in range(count):
        print(f"Capturing image {i+1}/{count} in 3 seconds...")
        time.sleep(3)
        
        ret, frame = cap.read()
        if ret:
            filename = f"{capture_dir}/capture_{i+1:02d}_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Saved: {filename}")
        else:
            print(f"Failed to capture image {i+1}")
        
        if i < count - 1:  # Don't wait after last image
            time.sleep(interval)
    
    cap.release()
    print(f"All images saved in '{capture_dir}' directory")
    return True

def main():
    print("Raspberry Pi Camera Test with OpenCV")
    print("====================================")
    
    # Check if camera is available
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: No camera detected!")
        print("Troubleshooting tips:")
        print("1. Check camera connection")
        print("2. Enable camera in raspi-config")
        print("3. Try different camera index (0, 1, 2)")
        print("4. Check if camera is being used by another process")
        return
    cap.release()
    
    while True:
        print("\nSelect test option:")
        print("1. Basic camera test")
        print("2. Camera with effects")
        print("3. Motion detection")
        print("4. Capture multiple images")
        print("5. Camera information")
        print("0. Exit")
        
        choice = input("Enter choice (0-5): ").strip()
        
        if choice == '1':
            test_camera_basic()
        elif choice == '2':
            test_camera_with_effects()
        elif choice == '3':
            test_motion_detection()
        elif choice == '4':
            count = int(input("Number of images to capture (default 5): ") or "5")
            interval = int(input("Interval between captures in seconds (default 2): ") or "2")
            capture_images(count, interval)
        elif choice == '5':
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                print_camera_info(cap)
                cap.release()
            else:
                print("Could not open camera for info")
        elif choice == '0':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()