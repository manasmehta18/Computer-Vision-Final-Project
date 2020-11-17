#! /usr/bin/env python3
import cv2 as cv
import os
import sys

def scale(src_dir, res_dir, size):
    print(src_dir)
    counter = 0
    for file in os.listdir(src_dir):
        print(file)
        img = cv.imread(os.path.join(src_dir, file))
        if img is None:
            sys.exit('Could not read the image: ' + file)
        new_file = cv.resize(img, size, 0, 0)
        cv.imwrite(os.path.join(res_dir, str(counter)+'.jpg'), new_file)
        cv.imshow('OpenCV Display Window', new_file)
        counter += 1

if __name__ == "__main__":
    scale(sys.argv[1], (64, 64))
