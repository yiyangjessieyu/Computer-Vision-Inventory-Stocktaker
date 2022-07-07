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


def recalculate(result_count, correct_count):
    ratio = result_count/correct_count
    if ratio < 0.5 or ratio > 2:
        process("ADAPT")
    else:
        print ("DONE")


def process(contour_method):

    # [contours]
    contour_dark = extract_contours(SOURCE_IMAGE, contour_method)
    show_wait_destroy("contour_dark", contour_dark)

    # [hough lines]
    hough, hough_dark = houghP(contour_dark)
    show_wait_destroy("hough_dark", hough_dark)

    # [count]
    result_count = count_houghNormal(hough_dark)
    print("HOUGH DARK WITHOUT DILATION. Inventory count of: " + str(result_count))

    # [dilate]
    dilate = dilation(hough_dark)
    show_wait_destroy("dilate", dilate)

    # Save the dark drawing of lines onto desktop.
    cv.imwrite(LOCAL_PATH + "dilate.png", dilate)

    return dilate


def main():
    data = {}

    # [load_image]
    global SOURCE_IMAGE
    SOURCE_IMAGE = read_image(LOCAL_PATH + INPUT_IMAGE_PATH)

    dilate = process("OTSU")

    # [count]
    result_count = count_houghNormal(dilate)
    print("DILATION. Inventory count of: " + str(result_count))

    # [data]
    # Comment out if you don't want to see accuracy of results.
    correct_count = get_correct_count(INPUT_IMAGE_PATH)
    data[INPUT_IMAGE_PATH] = (correct_count, result_count)
    output_results(data)

    recalculate(result_count, correct_count)



if __name__ == "__main__":
    main()
