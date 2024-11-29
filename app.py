from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load the trained model
model = load_model('signlanguagedetectionmodel48x48.h5')
labels = ['A', 'B', 'C', 'ErriPuvva', 'Delete']

# Predict function
def predict_sign(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    image = cv2.resize(image, (48, 48)) / 255.0
    image = np.expand_dims(image, axis=-1)  # Add channel dimension
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    predictions = model.predict(image)
    predicted_class = np.argmax(predictions, axis=1)[0]
    return labels[predicted_class]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Decode image from the request
        file = request.files['file']
        npimg = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        # Predict the sign
        result = predict_sign(img)
        return jsonify({'prediction': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
