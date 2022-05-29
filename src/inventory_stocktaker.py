# inventory_stocktaker.py
# potentially refine by using contours and ignoring curves
# find hough lines
# potentially refine
# make thicker
# potentially refine
# count hough lines

# Imports needed for this program to run.
import cv2 as cv
import numpy
from window import *
from file import *
from hough_line import *
from contours import *
from counter import *

def main():
    results = {}

    # [load_image]
    global SOURCE_IMAGE
    SOURCE_IMAGE = read_image(LOCAL_PATH + INPUT_IMAGE_PATH)
    show_wait_destroy("SOURCE_IMAGE", SOURCE_IMAGE)

    # [contours]
    contour = extract_contours(SOURCE_IMAGE)
    show_wait_destroy("contour", contour[1])
    show_wait_destroy("contour_dark", contour[2])

    # [hough lines]
    dark = houghNormal(contour)
    show_wait_destroy("dark", dark)



if __name__ == "__main__":
    main()
