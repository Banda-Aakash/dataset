import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Load preprocessed data from the NPZ file
PROCESSED_DATA_FILE = 'processed_data.npz'

def load_data():
    data = np.load(PROCESSED_DATA_FILE)
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    classes = data['classes']
    return X_train, X_test, y_train, y_test, classes

# Define the CNN model
def create_cnn_model(input_shape, num_classes):
    model = models.Sequential()
    
    # Convolutional Layer 1
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Convolutional Layer 2
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Convolutional Layer 3
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Flatten the output
    model.add(layers.Flatten())
    
    # Fully connected layer
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))  # Dropout to prevent overfitting
    
    # Output layer (number of classes = number of signs)
    model.add(layers.Dense(num_classes, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Main flow
if __name__ == "__main__":
    # Load the preprocessed data
    X_train, X_test, y_train, y_test, classes = load_data()

    # Input shape and number of classes
    input_shape = X_train.shape[1:]  # (64, 64, 3)
    num_classes = len(classes)

    # Create the CNN model
    model = create_cnn_model(input_shape, num_classes)

    # Train the model
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

    # Save the model
    model.save("sign_language_model.h5")
    print("Model saved to sign_language_model.h5")

    # Evaluate the model
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_acc:.4f}")
