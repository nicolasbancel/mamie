from __future__ import print_function
import cv2 as cv
import argparse
import os

######################################################
# Should simply name the file like "mamie0037.jpg"

# python3 canny_detector_tutorial.py --input mamie0037.jpg
######################################################


max_lowThreshold = 100
window_name = "Edge Map"
title_trackbar = "Min Threshold:"
ratio = 3
kernel_size = 3

PROCESSING_DIR = "/Users/nicolasbancel/git/perso/mamie/project/images/processing/canny_tutorial/"


def CannyThreshold(val):
    low_threshold = val
    img_blur = cv.blur(src_gray, (3, 3))
    detected_edges = cv.Canny(img_blur, low_threshold, low_threshold * ratio, kernel_size)
    mask = detected_edges != 0
    dst = src * (mask[:, :, None].astype(src.dtype))
    cv.imshow(window_name, dst)
    cv.imshow("Edges", detected_edges)
    print(f"{PROCESSING_DIR}canny_edges_lowthresh_{val}.jpg")
    cv.imwrite(f"{PROCESSING_DIR}canny_edges_lowthresh_{val}_{args.input}", detected_edges)


parser = argparse.ArgumentParser(description="Code for Canny Edge Detector tutorial.")
parser.add_argument("--input", help="Path to input image.", default="fruits.jpg")
args = parser.parse_args()

# Section below written custom way
MOSAIC_DIR = "/Users/nicolasbancel/git/perso/mamie/data/mosaic/"
file_path = os.path.join(MOSAIC_DIR, args.input)
src = cv.imread(file_path)
# End of custom

# src = cv.imread(cv.samples.findFile(args.input))
if src is None:
    print("Could not open or find the image: ", args.input)
    exit(0)
src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
cv.namedWindow(window_name)
cv.createTrackbar(title_trackbar, window_name, 0, max_lowThreshold, CannyThreshold)
CannyThreshold(0)
cv.waitKey()
