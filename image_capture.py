import cv2
import os
import time
from datetime import datetime

# Create a folder to save images
save_folder = "captured_images"
os.makedirs(save_folder, exist_ok=True)

# Initialize the camera
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

# Enable autofocus and set manual exposure
cap.set(cv2.CAP_PROP_SETTINGS, 1)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # Enable autofocus
exposure_value = -11
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Set manual exposure mode
cap.set(cv2.CAP_PROP_EXPOSURE, exposure_value)  # Adjust exposure manually
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 4656)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 3496)

# Ask for Box ID
boxid = input("Enter Box ID: ").strip()
count = 0
capturing = False

while True:
    ret, frame = cap.read()
    display_frame = cv2.resize(frame, (800, 600))
    if not ret:
        print("Failed to grab frame")
        break
    
    # Show the camera feed
    cv2.imshow("Camera", display_frame)
    count=count+1
    
    # Start or stop continuous capture when 'Enter' or 'Space' is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' ') or key == 13:
        capturing = not capturing
        print("Capturing started" if capturing else "Capturing stopped")
    
    if capturing:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Get current date-time
        img_name = os.path.join(save_folder, f"{boxid}_{timestamp}_{count}.jpg")
        cv2.imwrite(img_name, frame)
        print(f"Saved: {img_name}")
        time.sleep(0.2)  # Delay of 200ms
    
    # Adjust exposure with arrow keys
    elif key == ord('d'):
        exposure_value = min(exposure_value + 1, 0)
        cap.set(cv2.CAP_PROP_EXPOSURE, exposure_value)
        print(f"Exposure increased: {exposure_value}")
    elif key == ord('a'):
        exposure_value = max(exposure_value - 1, -10)
        cap.set(cv2.CAP_PROP_EXPOSURE, exposure_value)
        print(f"Exposure decreased: {exposure_value}")
    
    # Exit when 'Esc' is pressed
    elif key == 27:
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()