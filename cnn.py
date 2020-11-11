import numpy as np
import os
import mnist
import cv2 as cv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import shutil

import svm
import scaling

import argparse
import tensorflow as tf
config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(
    per_process_gpu_memory_fraction=0.8)
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--process', action='store_true')
options = parser.parse_args()

lighthouse_image_dir = "l"
other_image_dir = "o"

# Scale the data if flag exists
if options.process:
    shutil.rmtree("./" + lighthouse_image_dir)
    shutil.rmtree("./" + other_image_dir)
    os.mkdir(lighthouse_image_dir)
    os.mkdir(other_image_dir)
    scaling.scale("lighthouses", lighthouse_image_dir, (64, 64))
    scaling.scale("365", other_image_dir, (64, 64))


# Read images in as grayscale images and store as numpy arrays
# Label the classes as 1 for lighthouse, 0 for other
images = []
y = []
for file in os.listdir(lighthouse_image_dir):
    image = cv.imread(lighthouse_image_dir + "/" + file, 0)
    images.append(image)
    y.append(1)

for file in os.listdir(other_image_dir):
    images.append(cv.imread(other_image_dir + "/" + file, cv.IMREAD_GRAYSCALE))
    y.append(0)

# Split training/testing data
X_train, X_test, y_train, y_test = train_test_split(images,
                                                    y,
                                                    test_size=0.2,
                                                    stratify=y)

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# Normalize the images.
X_train = (X_train / 255) - 0.5
X_test = (X_test / 255) - 0.5

# Reshape the images.
X_train = np.expand_dims(X_train, axis=3)
X_test = np.expand_dims(X_test, axis=3)


num_filters = 100
filter_size = 3
pool_size = 2

# Build the model.
model = Sequential([
    Conv2D(num_filters, filter_size, activation='relu', input_shape=(64, 64, 1)),
    MaxPooling2D(pool_size=pool_size),
    Flatten(),
    Dense(2, activation='softmax'),
])


# Compile the model.
model.compile(
    'adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'],

)

n = 10

# Train the model.
model.fit(
    X_train,
    to_categorical(y_train),
    epochs=30,
    validation_data=(X_test[n:], to_categorical(y_test[n:])),
)


# Save the model to disk.
model.save_weights('cnn.h5')

# Load the model from disk later using:
# model.load_weights('cnn.h5')

# Predict on the first 5 test images.
predictions = model.predict(X_test[:n])

# Print our model's predictions.

predictions = np.argmax(predictions, axis=1)

print(predictions)  # [7, 2, 1, 0, 4]

labels = y_test[:n]
# Check our predictions against the ground truths.
print(labels)  # [7, 2, 1, 0, 4]

numCorrect = 0
for i in range(n):
    if predictions[i] == labels[i]:
        numCorrect += 1

print((numCorrect / n) * 100)
