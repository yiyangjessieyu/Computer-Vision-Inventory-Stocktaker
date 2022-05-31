# inventory_stocktaker.py
# contours of stright lines and ignoring curves
# find hough lines
# potentially refine
# make thicker
# count hough lines

# Imports needed for this program to run.
import cv2 as cv
import numpy
from window import *
from file import *
from hough_line import *
from contours import *
from dilate import *

# from counter import *





def main():
    results = {}

    # [load_image]
    global SOURCE_IMAGE
    print(000000000000000)
    print(LOCAL_PATH + INPUT_IMAGE_PATH)
    SOURCE_IMAGE = read_image(LOCAL_PATH + INPUT_IMAGE_PATH)

    # [contours]
    thresh, contour, contour_dark = extract_contours(SOURCE_IMAGE)

    # [hough lines]
    hough, hough_dark = houghNormal(contour_dark)
    show_wait_destroy("hough_dark", hough_dark)

    # [dilate]
    dilate = dilation(hough_dark)
    show_wait_destroy("dilate", dilate)

    print(count_houghNormal(dilate))


if __name__ == "__main__":
    main()
