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

    # [contours] TODO: refine this
    thresh, contour, contour_dark = extract_contours(SOURCE_IMAGE)
    show_wait_destroy("contour", contour)
    show_wait_destroy("contour_dark", contour_dark)

    # [hough lines]
    contour_dark = read_image(LOCAL_PATH + "contour_dark")
    show_wait_destroy("contour_dark222", contour_dark)
    dark = houghNormal(contour)
    show_wait_destroy("dark", dark)
    # Save the dark drawing of lines onto desktop.
    cv.imwrite(LOCAL_PATH, dark)

    # [erode]
    contour_dark = read_image(LOCAL_PATH + str(dark))
    erode = houghNormal(dark)
    show_wait_destroy("erode", erode)
    # Save the dark drawing of lines onto desktop.
    cv.imwrite(LOCAL_PATH, erode)

    print(erode)



if __name__ == "__main__":
    main()
