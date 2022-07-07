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

CONTOUR_METHODS = ["NORMAL", "OTSU", "ADAPT"]
HOUGH_METHODS = ["P", "NORMAL"]


def get_correct_count(input_image_path):
    count = ''
    for char in input_image_path:
        if char.isdigit():
            count += char
    return int(count)


def process(contour_method, hough_method):

    data = {}

    # [contours]
    contour_dark = extract_contours(SOURCE_IMAGE, contour_method)
    show_wait_destroy("contour_dark", contour_dark)

    # [hough lines]
    if hough_method == "NORMAL":
        hough, hough_dark = houghNormal(contour_dark)
    elif hough_method == "P":
        hough, hough_dark = houghP(contour_dark)
    show_wait_destroy("hough_dark", hough_dark)

    # [count]
    result_count = count_houghNormal(hough_dark)
    print("HOUGH DARK WITHOUT DILATION. Inventory count of: " + str(result_count))

    # [dilate]
    #dilate = dilation(hough_dark)
    #show_wait_destroy("dilate", dilate)

    # Save the dark drawing of lines onto desktop.
    #cv.imwrite(LOCAL_PATH + "dilate.png", dilate)

    dilate = hough_dark

    # [count]
    result_count = count_houghNormal(dilate) / 2
    print("DILATION. Inventory count of: " + str(result_count))

    # [data]
    # Comment out if you don't want to see accuracy of results.
    correct_count = get_correct_count(INPUT_IMAGE_PATH)
    data[INPUT_IMAGE_PATH] = (correct_count, result_count)
    output_results(data, contour_method, hough_method)

    return result_count


def main():

    # [load_image]
    global SOURCE_IMAGE
    SOURCE_IMAGE = read_image(LOCAL_PATH + INPUT_IMAGE_PATH)

    c_i = 0
    h_i = 0

    result_count = process(CONTOUR_METHODS[c_i], HOUGH_METHODS[h_i])

    while result_count < 5:
        if c_i >= len(CONTOUR_METHODS) - 1:
            h_i += 1
            c_i = 0
        else:
            c_i += 1
        result_count = process(CONTOUR_METHODS[c_i], HOUGH_METHODS[h_i])
    else:
        print ("DONE")


if __name__ == "__main__":
    main()
