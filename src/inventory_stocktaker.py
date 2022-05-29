# inventory_stocktaker.py
# find hough lines
# potentially refine
# make thicker
# potentially refine
# count hough lines

# Imports needed for this program to run.
import cv2 as cv
import numpy as np
from window import *
from file import *
from hough_line import *


def main():
    results = {}

    # [load_image]
    global SOURCE_IMAGE
    SOURCE_IMAGE = read_image(LOCAL_PATH + INPUT_IMAGE_PATH)
    show_wait_destroy("SOURCE_IMAGE", SOURCE_IMAGE)

    # [hough lines]
    dark = houghNormal(SOURCE_IMAGE)
    show_wait_destroy("dark", dark)


if __name__ == "__main__":
    main()
