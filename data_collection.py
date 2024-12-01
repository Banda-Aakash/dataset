import cv2
import os
import numpy as np

# Directory setup
directory = 'SignImage/'
if not os.path.exists(directory):
    os.mkdir(directory)

# Create directories for each letter and blank
letters = [chr(i) for i in range(65, 91)] + ['blank']
for letter in letters:
    letter_dir = os.path.join(directory, letter)
    if not os.path.exists(letter_dir):
        os.mkdir(letter_dir)

# Minimum value for thresholding
minValue = 70

# Preprocessing function
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 2)
    th3 = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    _, res = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return res

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Draw ROI on the frame
    cv2.rectangle(frame, (0, 40), (300, 300), (255, 255, 255), 2)
    cv2.imshow("Frame", frame)

    # Extract and preprocess ROI
    roi = frame[40:300, 0:300]
    processed_roi = preprocess_frame(roi)
    roi_resized = cv2.resize(processed_roi, (128, 128))

    # Display processed ROI
    cv2.imshow("Processed ROI", roi_resized)

    # Wait for user input
    interrupt = cv2.waitKey(10) & 0xFF

    # Map user input to directory
    if chr(interrupt).lower() in [chr(i).lower() for i in range(97, 123)] + ['.']:
        label = chr(interrupt).upper() if interrupt != ord('.') else 'blank'
        label_dir = os.path.join(directory, label)
        count = len(os.listdir(label_dir))
        save_path = os.path.join(label_dir, f"{label}_{count}.jpg")
        cv2.imwrite(save_path, roi_resized)
        print(f"Saved: {save_path}")

    # Quit on 'q'
    if interrupt == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
