#!/usr/bin/env python3
"""
Simple Raspberry Pi solenoid lock test script
Requires RPi.GPIO library: pip install RPi.GPIO
"""

import RPi.GPIO as GPIO
import time

# Configuration
SOLENOID_PIN = 18  # GPIO pin connected to relay/transistor controlling solenoid
LOCK_TIME = 2      # Time to keep solenoid activated (seconds)

def setup_gpio():
    """Initialize GPIO settings"""
    GPIO.setmode(GPIO.BCM)  # Use BCM pin numbering
    GPIO.setup(SOLENOID_PIN, GPIO.OUT)
    GPIO.output(SOLENOID_PIN, GPIO.LOW)  # Start with solenoid off
    print(f"GPIO pin {SOLENOID_PIN} configured for solenoid control")

def unlock_solenoid():
    """Activate solenoid to unlock"""
    print("Unlocking solenoid...")
    GPIO.output(SOLENOID_PIN, GPIO.HIGH)
    time.sleep(LOCK_TIME)
    GPIO.output(SOLENOID_PIN, GPIO.LOW)
    print("Solenoid locked")

def test_cycle():
    """Run a test cycle of unlock/lock"""
    try:
        setup_gpio()
        
        while True:
            print("\n--- Solenoid Test Menu ---")
            print("1. Unlock solenoid")
            print("2. Continuous test (5 cycles)")
            print("3. Exit")
            
            choice = input("Enter choice (1-3): ").strip()
            
            if choice == '1':
                unlock_solenoid()
                
            elif choice == '2':
                print("Running 5 test cycles...")
                for i in range(5):
                    print(f"Cycle {i+1}/5")
                    unlock_solenoid()
                    time.sleep(1)  # Wait between cycles
                print("Test cycles complete")
                
            elif choice == '3':
                break
                
            else:
                print("Invalid choice, please try again")
                
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        
    finally:
        GPIO.cleanup()
        print("GPIO cleanup complete")

if __name__ == "__main__":
    print("Raspberry Pi Solenoid Lock Test")
    print("WARNING: Ensure proper wiring and power supply!")
    print("Solenoid should be connected via relay or transistor")
    
    test_cycle()