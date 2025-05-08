import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2
import os
import pandas as pd
from sklearn.model_selection import train_test_split


# Define model architecture
def create_model(input_shape):
    model = models.Sequential()

    # Feature extraction
    model.add(layers.Conv2D(24, (5, 5), strides=(2, 2), activation='relu', input_shape=input_shape))
    model.add(layers.Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
    model.add(layers.Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())

    # Fully connected layers
    model.add(layers.Dense(100, activation='relu'))
    model.add(layers.Dense(50, activation='relu'))
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(1))  # Steering angle output

    model.compile(optimizer='adam', loss='mse')
    return model


# Load and prepare data
def prepare_data(data_dir, steering_file):
    # Read steering data
    steering_data = pd.read_csv(steering_file, header=None, names=['image', 'steering'])

    # Lists to store data
    X = []
    y = []

    # Load and preprocess images
    for index, row in steering_data.iterrows():
        img_path = os.path.join(data_dir, row['image'])
        if os.path.exists(img_path):
            # Load and preprocess image
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (200, 66))  # NVIDIA model input size
            img = img / 255.0  # Normalize

            X.append(img)
            y.append(row['steering'])

    return np.array(X), np.array(y)


# Train model
def train_model():
    # Load data
    X, y = prepare_data('output/training_data', 'output/training_data/steering_data.txt')

    # Split data
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create model
    model = create_model((66, 200, 3))  # Height, width, channels

    # Train model
    model.fit(X_train, y_train,
              validation_data=(X_valid, y_valid),
              epochs=10,
              batch_size=32)

    # Save model
    model.save('models/self_driving_model.h5')
    print("Model saved")


if __name__ == '__main__':
    train_model()