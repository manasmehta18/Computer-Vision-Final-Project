import numpy as np
import os
import mnist
import cv2 as cv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import EarlyStopping
import shutil
import sklearn as sk
import csv

import svm
import scaling
import augment
import config as cfg

import argparse
import tensorflow as tf

import time

config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(
    per_process_gpu_memory_fraction=0.8)
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--process', action='store_true')
parser.add_argument('-a', '--augment', action='store_true')
parser.add_argument('--pa', action='store_true')
options = parser.parse_args()

lighthouse_image_dir = "l"
other_image_dir = "o"

# Scale the data if flag exists
if options.process:
    shutil.rmtree("./" + lighthouse_image_dir)
    shutil.rmtree("./" + other_image_dir)
    os.mkdir(lighthouse_image_dir)
    os.mkdir(other_image_dir)
    scaling.scale("lighthouses", lighthouse_image_dir, (cfg.size[0], cfg.size[1]))
    scaling.scale("365", other_image_dir, (cfg.size[0], cfg.size[1]))


# Read images in as grayscale images and store as numpy arrays
# Label the classes as 1 for lighthouse, 0 for other
images = []
y = []
for file in os.listdir(lighthouse_image_dir):
    image = cv.imread(lighthouse_image_dir + "/" + file, 1 if cfg.rgb else 0)
    images.append(image)
    
    y.append(1)

for file in os.listdir(other_image_dir):
    image = cv.imread(other_image_dir + "/" + file, 1 if cfg.rgb else 0)
    images.append(image)
    y.append(0)

# Stratified KFold
kf = StratifiedKFold(n_splits=10, shuffle=True)
index = 1

acc_per_fold = []
loss_per_fold = []

results = []

for epochs in range(cfg.epochs, cfg.epochs + 1):
    for train_index, test_index in kf.split(images, y):
        start = time.time()

        images = np.array(images)
        y = np.array(y)

        X_train, X_test = images[train_index], images[test_index]
        y_train, y_test = y[train_index], y[test_index]

        if options.augment:
            aug_X_train = []
            aug_y_train = []
            for i, img in enumerate(X_train):
                augment_img = np.copy(img)
                augment_imgs = augment.augment(augment_img, options)

                if not os.path.exists('augment_out'):
                    os.makedirs('augment_out')

                if cfg.aug_imgs:  
                    from random import random
                    rand_str = str(random()).replace(".", "")
                    os.makedirs('augment_out/' + rand_str)
                    cv.imwrite('augment_out/' + rand_str + '/before.jpg', img) 
                    cv.imwrite('augment_out/' + rand_str + '/after.jpg', augment_imgs[0]) 
                    

                if options.pa:
                    X_train[i] = cv.Canny(X_train[i] ,100,200)


                # cv.imshow('image', augment_imgs[0])
                # cv.waitKey(0)
                # print(augment_imgs[0].shape)
                # print(X_train[i].shape)

                
                aug_X_train = aug_X_train + augment_imgs
                aug_y_train = aug_y_train + [ y_train[i] for _ in range(len(augment_imgs))] 
            if cfg.aug_imgs: 
                exit(0)
                
            X_train = np.concatenate((X_train, np.array(aug_X_train)))
            y_train = np.concatenate((y_train, np.array(aug_y_train)))

        if options.pa:
            for i in range(len(X_test)):
                X_test[i] = cv.Canny(X_test[i] ,100,200)
            

        # Normalize the images
        X_train = (X_train / 255.0) - 0.5
        X_test = (X_test / 255.0) - 0.5

        # Reshape the images.
        if not cfg.rgb:
            X_train = np.expand_dims(X_train, axis=3)
            X_test = np.expand_dims(X_test, axis=3)

        end = time.time()
        time_to_augment = str(end-start)
        start = time.time()

        #num_filters = 3      
        # Build the model.

        model = cfg.layer()

        # Compile the model.
        model.compile(
            'adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Generate a print
        print('--------------------------------')
        print('Split: {}'.format(index))

        # Train the model.
        model.fit(
            X_train,
            to_categorical(y_train),
            # y_train,
            epochs=epochs,
            validation_split=0.2
        )

    #    # Generate generalization metrics
    #    score = model.evaluate(X_test, y_test, verbose=0)
    #    print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
        # Generate generalization metrics
        scores = model.evaluate(X_test, to_categorical(y_test), verbose=0)

        end = time.time()
        time_to_train = str(end-start)
        
        print(
            f'Score for fold {index}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
        acc_per_fold.append(scores[1] * 100)
        loss_per_fold.append(scores[0])
        index += 1

        # HERE
        predicted_prob = model.predict(X_test) 
        target_predicted = np.argmax(predicted_prob, axis = 1)
        predicted_prob = predicted_prob.T[1].T
        target_true = y_test



        print(target_predicted)
        print(target_true)

        if cfg.do_images:
            for i, prob in enumerate(target_predicted):
                
                X_test[i] = 255 * (X_test[i] - X_test[i].min()) / (X_test[i].max() - X_test[i].min())
                X_test[i] = np.array(X_test[i], np.int)

                if target_predicted[i] == target_true[i]:
                    print("Hit! Lighthouse:", target_true[i], ", predicted_prob: ", predicted_prob[i])
                    if target_true[i] == 1:
                        cv.imwrite("data/TP/" + str(predicted_prob[i])+ ".jpg", X_test[i])
                    else:
                        cv.imwrite("data/TN/" + str(predicted_prob[i])+ ".jpg", X_test[i])
                else:
                    print("Miss! Lighthouse actually:", target_true[i], ", predicted_prob: ", predicted_prob[i])
                    if target_true[i] == 1:
                        cv.imwrite("data/FN/" + str(predicted_prob[i]) + ".jpg", X_test[i])
                    else:
                        cv.imwrite("data/FP/" + str(predicted_prob[i])+ ".jpg", X_test[i])
                
            cv.imshow('image', X_test[0])
            cv.waitKey(0)
        # # Takes 2 lists of length n

        #Precision
        precision = sk.metrics.precision_score(target_true, target_predicted,  average = 'binary')
        #Recall
        recall = sk.metrics.recall_score(target_true, target_predicted, average = 'binary')
        #ROC AUC
        roc_auc = sk.metrics.roc_auc_score(target_true, predicted_prob)
        #Confusion Matrix
        confusion_mtx = sk.metrics.confusion_matrix(target_true, target_predicted)
        #print('Confusion Matrix:\n{}'.format(confusion_mtx))
        tn, fp, fn, tp = confusion_mtx.ravel()

        f_score = sk.metrics.f1_score(target_true, target_predicted, average='micro')

        r = dict()
        r["precision"] = precision
        r["recall"] = recall
        r["roc"] = roc_auc
        r["confusion_matrix"] = str(confusion_mtx).replace("\n", ",")
        r["f_score"] = f_score
        r["time to augment (seconds)"] = time_to_augment
        r["time_to_train (seconds)"] = time_to_train
        r["loss_per_fold"] = scores[0]
        r["acc_per_fold"] = scores[1] * 100
    
        results.append(r)
    
    # Split training/testing data
    # X_train, X_test, y_train, y_test = train_test_split(images,
    #                                                    y,
    #                                                    test_size=0.2,
    #                                                    stratify=y)
    # == Provide average scores ==

    print('------------------------------------------------------------------------')
    print('Score per fold')
    for i in range(0, len(acc_per_fold)):
        print('------------------------------------------------------------------------')
        print(
            f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
        print(
        f'> Other metrics: {results[i]}')

    print('------------------------------------------------------------------------')
    print('Average scores for all folds:')
    print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
    print(f'> Loss: {np.mean(loss_per_fold)}')
    print('------------------------------------------------------------------------')

    keys = results[0].keys()
    with open('results.csv', 'w', newline='')  as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(results)
        

    # reset for next loop
    kf = StratifiedKFold(n_splits=10, shuffle=True)
    index = 1
    acc_per_fold = []
    loss_per_fold = []
    results = []

n = 10

# Save the model to disk.
# model.save_weights('cnn.h5')

# Load the model from disk later using:
# model.load_weights('cnn.h5')

# Predict on the first 5 test images.
#predictions = model.predict(X_test[:n])
#
# Print our model's predictions.
#
#predictions = np.argmax(predictions, axis=1)
#
# print(predictions)  # [7, 2, 1, 0, 4]
#
#labels = y_test[:n]
# Check our predictions against the ground truths.
# print(labels)  # [7, 2, 1, 0, 4]
#
#numCorrect = 0
# for i in range(n):
#    if predictions[i] == labels[i]:
#        numCorrect += 1
#
#print((numCorrect / n) * 100)
