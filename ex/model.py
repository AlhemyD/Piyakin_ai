import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split

def load_data(dataset_path, shapes=['circle', 'square', 'triangle'], img_size=(128, 128)):
    images = []
    labels = []

    for shape in shapes:
        image_dir = os.path.join(dataset_path, shape, 'images')
        label_dir = os.path.join(dataset_path, shape, 'labels')

        for img_file in os.listdir(image_dir):
            img_path = os.path.join(image_dir, img_file)
            label_path = os.path.join(label_dir, img_file.replace('.png', '.txt'))

            # Читаем изображение
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, img_size)
            images.append(img)

            # Читаем разметку (Roboflow YOLO format: class x_center y_center width height)
            with open(label_path, 'r') as f:
                line = f.readline().strip().split()
                bbox = [float(x) for x in line[1:]]  # [x_center, y_center, width, height]
                labels.append(bbox)

    images = np.array(images).reshape(-1, img_size[0], img_size[1], 1) / 255.0
    labels = np.array(labels)
    return images, labels
def split_data(images, labels, test_size=0.2):
    return train_test_split(images, labels, test_size=test_size, random_state=42)

dataset_path = "dataset"
images, labels = load_data(dataset_path)
X_train, X_test, y_train, y_test = split_data(images, labels)
print(f"Train images: {len(X_train)}, Test images: {len(X_test)}")

def create_model(input_shape=(128, 128, 1)):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(4, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train_model(model, X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test),epochs=epochs, batch_size=batch_size, verbose=1)
    return history

def save_model(model, path="shape_detector.keras"):
    model.save(path)

images, labels = load_data(dataset_path)
X_train, X_test, y_train, y_test = split_data(images, labels)

model = create_model()
train_model(model, X_train, y_train, X_test, y_test)
save_model(model)