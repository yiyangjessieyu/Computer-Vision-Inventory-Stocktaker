# inventory_stocktaker.py
# contours of stright lines and ignoring curves
# find hough lines
# make thicker
# count hough lines

# Imports needed for this program to run.
import re
import cv2 as cv
import numpy
from window import *
from file import *
from hough_line import *
from contours import *
from dilate import *


# from counter import *

def get_correct_count(input_image_path):
    count = ''
    for char in input_image_path:
        if char.isdigit():
            count += char
    return int(count)


def main():
    data = {}

    # [load_image]
    global SOURCE_IMAGE
    SOURCE_IMAGE = read_image(LOCAL_PATH + INPUT_IMAGE_PATH)

    # [contours]
    thresh, contour, contour_dark, contour_dark2, otsuThreshInv = extract_contours(SOURCE_IMAGE)
    show_wait_destroy("contour_dark", contour_dark)
    show_wait_destroy("otsuThreshInv", otsuThreshInv)

    adaptive_thresh = read_image(LOCAL_PATH + 'saved_image.jpg')

    # [hough lines]
    hough, hough_dark = houghNormal(contour_dark2)
    show_wait_destroy("hough_dark", hough_dark)

    # [dilate]
    dilate = dilation(hough_dark)
    show_wait_destroy("dilate", dilate)

    # Save the dark drawing of lines onto desktop.
    cv.imwrite(LOCAL_PATH + "dilate.png", dilate)

    # [count]
    result_count = count_houghNormal(dilate)
    print("Inventory count of: " + str(result_count))

    # [data]
    # Comment out if you don't want to see accuracy of results.
    correct_count = get_correct_count(INPUT_IMAGE_PATH)
    data[INPUT_IMAGE_PATH] = (correct_count, result_count)
    output_results(data)



if __name__ == "__main__":
    main()
