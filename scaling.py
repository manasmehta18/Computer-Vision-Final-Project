#! /usr/bin/env python3
import cv2 as cv
import os
import sys
import config as cfg


def crop_and_resize(img, w, h):
    if cfg.rgb and not cfg.canny:
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
        if cfg.rgb and not cfg.canny:
            img = img[:, x1:x2, :]
        else:
            img = img[:, x1:x2]
    if input_aspect_ratio < res_aspect_ratio:
        im_w_r = w
        im_h_r = int(w/input_aspect_ratio)
        img = cv.resize(img, (im_w_r, im_h_r))
        y1 = int((im_h_r - h)/2)
        y2 = y1 + h
        if cfg.rgb and not cfg.canny:
            img = img[y1:y2, :, :]
        else:
            img = img[y1:y2, :]
    if input_aspect_ratio == res_aspect_ratio:
        img = cv.resize(img, (w, h))

    return img



def scale(src_dir, res_dir, size):
    print(src_dir)
    counter = 0
    for file in os.listdir(src_dir):
        print(file)
        img = cv.imread(os.path.join(src_dir, file), 1 if cfg.rgb else 0)
        if img is None:
            sys.exit('Could not read the image: ' + file)

        if cfg.canny:
            img = cv.Canny(img,100,200)

        new_file = crop_and_resize(img, cfg.size[0], cfg.size[1])
        cv.imwrite(os.path.join(res_dir, str(counter)+'.jpg'), new_file)
        counter += 1

if __name__ == "__main__":
    scale(sys.argv[1], (64, 64))
