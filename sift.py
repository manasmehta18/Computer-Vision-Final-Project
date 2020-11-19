#! /usr/bin/env python3
import cv2 as cv
import matplotlib.pyplot as plt
#%matplotlib inline

def sift(lighthouse, detect_lighthouse, blur = False):
    model_img = cv.imread(lighthouse)
    model_gray = cv.cvtColor(model_img, cv.COLOR_BGR2GRAY)

    if blur:
        model_img = cv.GaussianBlur(model_img, (5, 5), 0)
        model_gray = cv.GaussianBlur(model_gray, (5, 5), 0)

    # keypoints
    sift = cv.SIFT_create()
    keypoints = sift.detect(model_gray, None)

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
    kp, des = sift.detectAndCompute(model_gray, None)

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

    match_img = cv.drawMatches(model_gray, kp, detect_gray, kp2, matches[:20], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv.imwrite('Matches.jpg', match_img)
    #plt.imshow(match_img),plt.show()

if __name__ == "__main__":
    #sift('pigeonpointlighthouse.jpg', 'PigeonPoint/Pigeon-Point-Vertical.jpg', True)
    sift('pigeonpointlighthouse.jpg', 'Random/CapePalliserNZ.jpg', True)
