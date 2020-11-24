#! /usr/bin/env python3
import cv2 as cv
import matplotlib.pyplot as plt
import os
import numpy as np

MIN_MATCH_COUNT = 10

def sift(lighthouse, detect_lighthouse, counter, mask=False):
    model_img = cv.imread(lighthouse)
    model_gray = cv.cvtColor(model_img, cv.COLOR_BGR2GRAY)

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
    image_name += '.jpg'
    cv.imwrite(image_name, model_img_kp)
    cv.imshow('OpenCV Display Window', model_img_kp)

    # keypoints and descriptors
    kp, des = sift.detectAndCompute(model_gray, mask=model_mask)

    # Read in image to compare
    detect_img = cv.imread(detect_lighthouse)
    detect_gray = cv.cvtColor(detect_img, cv.COLOR_BGR2GRAY)

    keypoints2 = sift.detect(detect_gray, None)
    detect_img_kp = cv.drawKeypoints(detect_gray, keypoints2, None)

    kp2, des2 = sift.detectAndCompute(detect_gray, None)
    cv.imwrite('keypoints_detection.jpg', detect_img_kp)
    
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des,des2,k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()
        h,w = model_gray.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv.perspectiveTransform(pts,M)
        detect_gray = cv.polylines(detect_gray,[np.int32(dst)],True,255,3, cv.LINE_AA)
    else:
        print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
        matchesMask = None

    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                       singlePointColor = None,
                       matchesMask = matchesMask, # draw only inliers
                       flags = 2)
    match_img = cv.drawMatches(model_gray,kp,detect_gray,kp2,good,None,**draw_params)
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
        sift('pigeonpointlighthouse.jpg', 'Random/{}'.format(image), counter, True)
        counter += 1
    pigeonpoint_images = os.listdir('PigeonPoint')
    pigeonpoint_images.sort()
    print(pigeonpoint_images)
    for image in pigeonpoint_images:
        sift('pigeonpointlighthouse.jpg', 'PigeonPoint/{}'.format(image), counter, True)
        counter += 1
