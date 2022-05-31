# inventory_stocktaker.py
# potentially refine by using contours and ignoring curves
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


# from counter import *


def transform_grey(img):
    # Transform source image to gray if it is not already
    if len(img.shape) != 2:
        return cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    else:
        return img


def transform_bitwise(img):
    # Apply adaptiveThreshold at the bitwise_not of gray, notice the ~ symbol
    bitwise = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 15, -2)
    return bitwise


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

    # [gray]
    gray = transform_grey(hough_dark)
    show_wait_destroy("gray", gray)

    # [bitwise]
    bitwise = transform_bitwise(gray)
    show_wait_destroy("bitwise", bitwise)

    kernel = np.ones((4, 4), np.uint8)
    dilate = cv.dilate(bitwise, kernel, iterations=5)
    show_wait_destroy("dilate", dilate)

    print(count_houghNormal(dilate))


if __name__ == "__main__":
    main()
