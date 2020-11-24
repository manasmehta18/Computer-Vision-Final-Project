#! /usr/bin/env python3
import cv2 as cv
import matplotlib.pyplot as plt
import os
import numpy as np

def sift(lighthouse, detect_lighthouse, counter, blur = False, mask=False):
    model_img = cv.imread(lighthouse)
    model_gray = cv.cvtColor(model_img, cv.COLOR_BGR2GRAY)

    if blur:
        model_img = cv.GaussianBlur(model_img, (5, 5), 0)
        model_gray = cv.GaussianBlur(model_gray, (5, 5), 0)

    # keypoints
    sift = cv.SIFT_create()

    # Create Masks
    model_mask = np.ones(model_gray.shape)
    if mask:
        model_mask[:, :int(model_mask.shape[1]* 2/10)] = 0
        model_mask[:, int(model_mask.shape[1]* 7/10):] = 0
        model_mask[:int(model_mask.shape[0]* 1/20), :] = 0
        model_mask[int(model_mask.shape[0]* 17/20):, :] = 0
    model_mask = model_mask.astype(np.uint8)
    if mask:
        masked_img = cv.bitwise_and(model_gray, model_gray, mask=model_mask)
        cv.imwrite('Masked Lighthouse.jpg', masked_img)

    keypoints = sift.detect(model_gray, mask=model_mask)

    model_img_kp = cv.drawKeypoints(model_gray, keypoints, None)
    #model_img_kp = cv.drawKeypoints(model_gray, keypoints, model_img, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Draw the keypoints
    image_name = 'sift_keypoints'
    if blur:
        image_name += '_blur'
    image_name += '.jpg'
    cv.imwrite(image_name, model_img_kp)
    cv.imshow('OpenCV Display Window', model_img_kp)

    # keypoints and descriptors
    kp, des = sift.detectAndCompute(model_gray, mask=model_mask)

    # Read in image to compare
    detect_img = cv.imread(detect_lighthouse)
    detect_gray = cv.cvtColor(detect_img, cv.COLOR_BGR2GRAY)

    if blur:
        detect_img = cv.GaussianBlur(detect_img, (5, 5), 0)
        detect_gray = cv.GaussianBlur(detect_gray, (5, 5), 0)

    keypoints2 = sift.detect(detect_gray, None)
    detect_img_kp = cv.drawKeypoints(detect_gray, keypoints2, None)

    kp2, des2 = sift.detectAndCompute(detect_gray, None)
    cv.imwrite('keypoints_detection.jpg', detect_img_kp)
    
    # Feature matching
    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
    
    matches = bf.match(des, des2)
    matches = sorted(matches, key = lambda x:x.distance)
    print(len(matches))

    match_img = cv.drawMatches(model_gray, kp, detect_gray, kp2, matches[:100], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv.imwrite('Matches{}.jpg'.format(counter), match_img)
    #plt.imshow(match_img),plt.show()

if __name__ == "__main__":
    #sift('pigeonpointlighthouse.jpg', 'Random/1.jpg', False)
    #sift('pigeonpointlighthouse.jpg', 'Random/2.jpg', False)
    #sift('pigeonpointlighthouse.jpg', 'Random/3.jpeg', False)
    #sift('pigeonpointlighthouse.jpg', 'Random/4.jpeg', False)
    #sift('pigeonpointlighthouse.jpg', 'Random/5.jpg', False)
    #sift('pigeonpointlighthouse.jpg', 'PigeonPoint/1.jpg', False)
    #sift('pigeonpointlighthouse.jpg', 'PigeonPoint/2.jpeg', False)
    #sift('pigeonpointlighthouse.jpg', 'PigeonPoint/3.jpg', False)
    #sift('pigeonpointlighthouse.jpg', 'PigeonPoint/4.gif', False) # Can't use .gif format
    #sift('pigeonpointlighthouse.jpg', 'PigeonPoint/5.jpg', False)
    #sift('pigeonpointlighthouse.jpg', 'PigeonPoint/6.jpg', False)
    #sift('pigeonpointlighthouse.jpg', 'PigeonPoint/7.jpg', False)
    #sift('pigeonpointlighthouse.jpg', 'PigeonPoint/8.jpg', False)
    #sift('pigeonpointlighthouse.jpg', 'PigeonPoint/9.jpg', False)
    counter = 1
    random_images = os.listdir('Random')
    random_images.sort()
    print(random_images)
    for image in random_images:
        sift('pigeonpointlighthouse.jpg', 'Random/{}'.format(image), counter, False, True)
        counter += 1
    pigeonpoint_images = os.listdir('PigeonPoint')
    pigeonpoint_images.sort()
    print(pigeonpoint_images)
    for image in pigeonpoint_images:
        sift('pigeonpointlighthouse.jpg', 'PigeonPoint/{}'.format(image), counter, False, True)
        counter += 1
