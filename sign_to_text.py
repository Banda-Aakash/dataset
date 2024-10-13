import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model
import tkinter as tk
from PIL import Image, ImageTk

# Load the trained model
model = load_model('sign_language_model.h5')
labels = ['A', 'B', 'C', 'D', ..., 'Z', 'Space', 'Delete']  # Define your labels

# Function to predict the sign
def predict_sign(image, model, labels):
    image = cv2.resize(image, (64, 64)) / 255.0
    image = np.expand_dims(image, axis=0)  # Reshape for model input
    
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions, axis=1)[0]
    predicted_label = labels[predicted_class]
    
    return predicted_label

class SignLanguageApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        
        self.video_source = 0
        self.vid = cv2.VideoCapture(self.video_source)

        self.canvas = tk.Canvas(window, width=640, height=480)
        self.canvas.pack()

        # Label to display predicted sign
        self.label = tk.Label(window, text="Predicted sign: ", font=("Helvetica", 16))
        self.label.pack()

        # Create buttons
        self.btn_snapshot = tk.Button(window, text="Capture Sign", width=20, command=self.snapshot)
        self.btn_snapshot.pack(side=tk.LEFT, padx=10)

        self.btn_clear = tk.Button(window, text="Clear All", width=20, command=self.clear_text)
        self.btn_clear.pack(side=tk.LEFT, padx=10)

        self.btn_quit = tk.Button(window, text="Quit", width=20, command=self.quit_app)
        self.btn_quit.pack(side=tk.LEFT, padx=10)

        self.textbox = tk.Text(window, height=5, width=52)
        self.textbox.pack(pady=10)

        # Start video loop
        self.delay = 10
        self.is_snapshot_mode = False
        self.update()
        self.window.mainloop()

    def update(self):
        ret, frame = self.vid.read()
        if ret:
            # Define region of interest (ROI) for the hand
            roi = frame[100:300, 100:300]
            
            # Snapshot effect: Draw a filled rectangle if snapshot mode is enabled
            if self.is_snapshot_mode:
                cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), -1)  # Green filled rectangle as snapshot
                time.sleep(0.2)  # Pause for visual feedback
                self.is_snapshot_mode = False  # Reset after snapshot effect

            # Draw ROI boundary
            cv2.rectangle(frame, (100, 100), (300, 300), (255, 255, 0), 2)

            # Convert image to RGB for Tkinter display
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)

            # Display the video feed on canvas
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            self.canvas.imgtk = imgtk

        self.window.after(self.delay, self.update)

    def snapshot(self):
        ret, frame = self.vid.read()
        if ret:
            roi = frame[100:300, 100:300]  # Crop the hand region
            
            # Predict sign from the ROI
            predicted_sign = predict_sign(roi, model, labels)
            
            # Update the predicted sign label
            self.label.config(text=f"Predicted sign: {predicted_sign}")
            
            # Update the text box with the prediction
            self.textbox.insert(tk.END, predicted_sign)

            # Activate snapshot effect
            self.is_snapshot_mode = True

    def clear_text(self):
        self.textbox.delete(1.0, tk.END)

    def quit_app(self):
        self.window.quit()
        self.vid.release()
        cv2.destroyAllWindows()

# Run the application
SignLanguageApp(tk.Tk(), "Sign Language to Text")
