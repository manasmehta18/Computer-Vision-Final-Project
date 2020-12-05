
#! /usr/bin/env python3
import cv2 as cv
import os
import sys
import random
import numpy as np
from scipy import ndimage

import config as cfg

# Define the translate function
def translate(image, x, y):
    # Define translation matrix
    M = np.float32([[1, 0, x], [0, 1, y]])
    shifted = cv.warpAffine(image, M, (image.shape[1], image.shape[0]))
    # Return the converted image
    return shifted

def rotate(image, theta):
    rotated = ndimage.rotate(image, theta)
    return rotated

def zoom_in(img, amt):
    if cfg.rgb:
        w, h , c = img.shape
    else: 
        w, h  = img.shape

    zh = int(np.ceil(h / amt))
    zw = int(np.ceil(w / amt))
    top = (h - zh) // 2
    left = (w - zw) // 2

    if cfg.rgb:
        out = ndimage.zoom(img[top:top+zh, left:left+zw, :], amt)
    else:
        out = ndimage.zoom(img[top:top+zh, left:left+zw], amt)

    trim_top = ((out.shape[0] - h) // 2)
    trim_left = ((out.shape[1] - w) // 2)

    if cfg.rgb:
        out = out[trim_top:trim_top+h, trim_left:trim_left+w, :]
    else:
        out = out[trim_top:trim_top+h, trim_left:trim_left+w]
    return out

def lighting(img, gamma):
    invGamma = 1.0 / gamma
    table = np.array([
        ((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)])
    return cv.LUT(img.astype(np.uint8), table.astype(np.uint8))

def crop_and_resize(img, w, h):
    if cfg.rgb:
        im_h, im_w, c = img.shape
    else: 
        im_h, im_w = img.shape
    res_aspect_ratio = w/h
    input_aspect_ratio = im_w/im_h

    if input_aspect_ratio > res_aspect_ratio:
        im_w_r = int(input_aspect_ratio*h)
        im_h_r = h
        img = cv.resize(img, (im_w_r, im_h_r))
        x1 = int((im_w_r - w)/2)
        x2 = x1 + w
        if cfg.rgb:
            img = img[:, x1:x2, :]
        else:
            img = img[:, x1:x2]
    if input_aspect_ratio < res_aspect_ratio:
        im_w_r = w
        im_h_r = int(w/input_aspect_ratio)
        img = cv.resize(img, (im_w_r, im_h_r))
        y1 = int((im_h_r - h)/2)
        y2 = y1 + h
        if cfg.rgb:
            img = img[y1:y2, :, :]
        else:
            img = img[y1:y2, :]
    if input_aspect_ratio == res_aspect_ratio:
        img = cv.resize(img, (w, h))
    return img


def augment(img, o):
    img = lighting(img, random.randint(50, 150) / 100.0)

    if o.pa:
        img = cv.Canny(img,100,200)

    img = translate(img, random.randint(-7, 7), random.randint(-7, 7))

    if random.randint(1, 2) == 1:
        # rotate left
        img = rotate(img, random.randint(3, 15))
    else: 
        img = rotate(img, random.randint(-15, -3))

    img = zoom_in(img, random.randint(11, 13) / 10.0)


    # print(img)

    img = crop_and_resize(img, cfg.size[0], cfg.size[1])

    if cfg.rgb:
        img = img[:,:,:3]
    
    # cv.imshow('image',img)
    # cv.waitKey(0)
    
    # cv.imshow('image',img)
    # cv.waitKey(0)

    return [img]
    