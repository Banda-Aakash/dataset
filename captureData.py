import cv2
import os
import numpy as np

# Directory where the images will be stored
DATASET_DIR = 'sign_language_dataset'
if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR)

# Function to capture images for different signs
def capture_images():
    cap = cv2.VideoCapture(0)  # Start video capture
    num_images = 200  # Number of images to capture for each sign
    print("Press 's' to start capturing images for a sign, and 'q' to quit.")
    
    while True:
        current_sign = input("Enter the sign label (e.g., A, B, C...): ").upper()
        if current_sign == 'Q':
            break
        
        sign_dir = os.path.join(DATASET_DIR, current_sign)
        if not os.path.exists(sign_dir):
            os.makedirs(sign_dir)
        
        image_count = 0
        while image_count < num_images:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Display the frame
            cv2.imshow(f"Capturing images for sign {current_sign}", frame)

            # Capture images when 's' is pressed
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                image_count += 1
                image_path = os.path.join(sign_dir, f"{current_sign}_{image_count}.jpg")
                cv2.imwrite(image_path, frame)
                print(f"Captured {image_path}")

            # Press 'q' to quit the capture
            if key == ord('q'):
                break
        
        print(f"Captured {image_count} images for sign {current_sign}.")

    cap.release()
    cv2.destroyAllWindows()

# Function to preprocess the images (resize and normalize)
def preprocess_images(dataset_dir, target_size=(64, 64)):
    image_data = []
    labels = []

    # Loop through each sign folder
    for sign_label in os.listdir(dataset_dir):
        sign_folder = os.path.join(dataset_dir, sign_label)
        
        # Loop through each image in the folder
        for img_file in os.listdir(sign_folder):
            img_path = os.path.join(sign_folder, img_file)
            image = cv2.imread(img_path)

            # Preprocess the image: resize and normalize
            image = cv2.resize(image, target_size)
            image = image / 255.0  # Normalize pixel values to [0, 1]
            image_data.append(image)
            labels.append(sign_label)
    
    # Convert lists to numpy arrays
    image_data = np.array(image_data)
    labels = np.array(labels)
    
    return image_data, labels

# Main flow
if __name__ == "__main__":
    capture_images()  # Step 1: Capture images to create dataset

    # Step 2: Preprocess images after capturing
    image_data, labels = preprocess_images(DATASET_DIR)

    print(f"Total images captured: {len(image_data)}")
    print(f"Example preprocessed image shape: {image_data[0].shape}")
