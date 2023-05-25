from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse

LOCAL_PATH = "/csse/users/yyu69/Desktop/COSC428/Project-april21/Computer-Vision-Inventory-Stocktaker/"
OUTPUT_IMAGE_PATH = './resources/dilatation_dst.png'
ERODE_OUTPUT_IMAGE_PATH = './resources/erode.png'


src = None
erosion_size = 0
max_elem = 2
max_kernel_size = 21
title_trackbar_element_shape = 'Element:\n 0: Rect \n 1: Cross \n 2: Ellipse'
title_trackbar_kernel_size = 'Kernel size:\n 2n +1'
title_erosion_window = 'Erosion Demo'
title_dilation_window = 'Dilation Demo'


def main(image):
    global src
    src = cv.imread(image)
    # Resize smaller
    src = cv.resize(src, (int(src.shape[1] * 0.7), int(src.shape[0] * 0.7)))
    if src is None:
        print('Could not open or find the image: ', image)
        exit(0)
    cv.namedWindow(title_erosion_window)
    cv.createTrackbar(title_trackbar_element_shape, title_erosion_window, 0, max_elem, erosion)
    cv.createTrackbar(title_trackbar_kernel_size, title_erosion_window, 0, max_kernel_size, erosion)
    cv.namedWindow(title_dilation_window)
    cv.createTrackbar(title_trackbar_element_shape, title_dilation_window, 0, max_elem, dilatation)
    cv.createTrackbar(title_trackbar_kernel_size, title_dilation_window, 0, max_kernel_size, dilatation)
    erosion(0)
    dilatation(0)
    cv.waitKey()


# optional mapping of values with morphological shapes
def morph_shape(val):
    if val == 0:
        return cv.MORPH_RECT
    elif val == 1:
        return cv.MORPH_CROSS
    elif val == 2:
        return cv.MORPH_ELLIPSE


def erosion(val):
    erosion_size = cv.getTrackbarPos(title_trackbar_kernel_size, title_erosion_window)
    erosion_shape = morph_shape(cv.getTrackbarPos(title_trackbar_element_shape, title_erosion_window))

    element = cv.getStructuringElement(erosion_shape, (2 * erosion_size + 1, 2 * erosion_size + 1),
                                       (erosion_size, erosion_size))

    erosion_dst = cv.erode(src, element)
    cv.imshow(title_erosion_window, erosion_dst)

    cv.imwrite(LOCAL_PATH + ERODE_OUTPUT_IMAGE_PATH, erosion_dst)


def dilatation(val):
    dilatation_size = cv.getTrackbarPos(title_trackbar_kernel_size, title_dilation_window)
    dilation_shape = morph_shape(cv.getTrackbarPos(title_trackbar_element_shape, title_dilation_window))
    element = cv.getStructuringElement(dilation_shape, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                       (dilatation_size, dilatation_size))
    dilatation_dst = cv.dilate(src, element)
    cv.imshow(title_dilation_window, dilatation_dst)

    cv.imwrite(LOCAL_PATH + OUTPUT_IMAGE_PATH, dilatation_dst)


image = "/csse/users/yyu69/Desktop/COSC428/Project-april21/Computer-Vision-Inventory-Stocktaker/resources/dark.png"
main(image)
