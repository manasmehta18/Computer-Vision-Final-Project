import numpy as np
import os
import mnist
import cv2 as cv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
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

# Stratified KFold
kf = StratifiedKFold(n_splits=10, shuffle=True)
index = 1

acc_per_fold = []
loss_per_fold = []

for train_index, test_index in kf.split(images, y):
    images = np.array(images)
    y = np.array(y)

    X_train, X_test = images[train_index], images[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Normalize the images.
    X_train = (X_train / 255) - 0.5
    X_test = (X_test / 255) - 0.5


    # Reshape the images.
    X_train = np.expand_dims(X_train, axis=3)
    X_test = np.expand_dims(X_test, axis=3)


    #num_filters = 3
    filter_size = 3
    pool_size = 2

    # Build the model.
    model = Sequential([
        Conv2D(10, filter_size, activation='relu', input_shape=(64, 64, 1)),
        MaxPooling2D(pool_size=pool_size),
        Conv2D(20, filter_size, activation='relu'),
        MaxPooling2D(pool_size=pool_size),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(2, activation='softmax'),
    ])


    # Compile the model.
    model.compile(
        'adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    # Generate a print
    print('--------------------------------')
    print('Split: {}'.format(index))

    # Train the model.
    model.fit(
        X_train,
        to_categorical(y_train),
        #y_train,
        epochs=15,
        #validation_data=(X_test[n:], to_categorical(y_test[n:])),
    )

#    # Generate generalization metrics
#    score = model.evaluate(X_test, y_test, verbose=0)
#    print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
    # Generate generalization metrics
    scores = model.evaluate(X_test, to_categorical(y_test), verbose=0)
    print(f'Score for fold {index}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])
    index += 1

# Split training/testing data
#X_train, X_test, y_train, y_test = train_test_split(images,
#                                                    y,
#                                                    test_size=0.2,
#                                                    stratify=y)

# == Provide average scores ==
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_per_fold)):
  print('------------------------------------------------------------------------')
  print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')
print('------------------------------------------------------------------------')


n = 10



# Save the model to disk.
model.save_weights('cnn.h5')

# Load the model from disk later using:
# model.load_weights('cnn.h5')

# Predict on the first 5 test images.
#predictions = model.predict(X_test[:n])
#
## Print our model's predictions.
#
#predictions = np.argmax(predictions, axis=1)
#
#print(predictions)  # [7, 2, 1, 0, 4]
#
#labels = y_test[:n]
## Check our predictions against the ground truths.
#print(labels)  # [7, 2, 1, 0, 4]
#
#numCorrect = 0
#for i in range(n):
#    if predictions[i] == labels[i]:
#        numCorrect += 1
#
#print((numCorrect / n) * 100)
