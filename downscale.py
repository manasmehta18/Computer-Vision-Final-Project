
import cv2
import numpy as np
import os


def crop_and_resize(img, w, h):
    im_h, im_w, channels = img.shape
    res_aspect_ratio = w/h
    input_aspect_ratio = im_w/im_h

    if input_aspect_ratio > res_aspect_ratio:
        im_w_r = int(input_aspect_ratio*h)
        im_h_r = h
        img = cv2.resize(img, (im_w_r, im_h_r))
        x1 = int((im_w_r - w)/2)
        x2 = x1 + w
        img = img[:, x1:x2, :]
    if input_aspect_ratio < res_aspect_ratio:
        im_w_r = w
        im_h_r = int(w/input_aspect_ratio)
        img = cv2.resize(img, (im_w_r, im_h_r))
        y1 = int((im_h_r - h)/2)
        y2 = y1 + h
        img = img[y1:y2, :, :]
    if input_aspect_ratio == res_aspect_ratio:
        img = cv2.resize(img, (w, h))

    return img


directory = "to_process"
for filename in os.listdir(directory):
    try:
        path = os.path.join(directory, filename)
        # Reading the image in RGB mode
        img = cv2.imread(path, 1)
        img = crop_and_resize(img, 400, 400)
        cv2.imwrite(os.path.join("processed", filename), img)
    except Exception:
        continue
