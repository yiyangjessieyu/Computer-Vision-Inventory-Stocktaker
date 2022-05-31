# Imports needed for this program to run.
import cv2 as cv
import numpy as np

from window import *

KERNEL_SIZE = 3
ITERATIONS = 3


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


def dilation(image):
    gray = transform_grey(image)
    bitwise = transform_bitwise(gray)

    kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)
    dilate = cv.dilate(bitwise, kernel, iterations=ITERATIONS)

    return dilate
