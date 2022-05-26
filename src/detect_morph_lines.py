"""
@file morph_lines_detection.py
@brief Use morphology transformations for extracting horizontal and vertical lines sample code
Tutorial here: https://docs.opencv.org/3.4/dd/dd7/tutorial_morph_lines_detection.html
"""
import numpy as np
import sys
import cv2 as cv

LOCAL_PATH = "/csse/users/yyu69/Desktop/COSC428/Project-april21/Computer-Vision-Inventory-Stocktaker/"
INPUT_IMAGE_PATH = 'resources/side_hearts.jpg'

def show_wait_destroy(winname, img):
    cv.imshow(winname, img)
    #cv.resizeWindow(winname, 50, 100)
    cv.moveWindow(winname, 500, 0)
    cv.waitKey(0)
    cv.destroyWindow(winname)

def show_wait_overlay(src, img):
    winname = 'Blended Image'
    dst = cv.addWeighted(src, 0.5, img, 0.7, 0)
    img_arr = np.hstack((src, img))
    cv.imshow('Input Images', img_arr)
    cv.imshow('Blended Image', dst)
    cv.moveWindow(winname, 500, 0)
    cv.waitKey(0)
    cv.destroyWindow(winname)


def read_image(argv):
    # Check number of arguments
    if len(argv) < 1:
        print ('Not enough parameters')
        print ('Usage:\nmorph_lines_detection.py < path_to_image >')
        return -1

    # Load the image
    src = cv.imread(argv[0], cv.IMREAD_COLOR)

    # Check if image is loaded fine
    if src is None:
        print ('Error opening image: ' + argv[0])
        return -1

    # Resize smaller
    src = cv.resize(src, (int(src.shape[1] * 0.7), int(src.shape[0] * 0.7)))

    return src

def transform_grey(src):
    # Transform source image to gray if it is not already
    if len(src.shape) != 2:
        return cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    else:
        return src

def main(argv):

    # [load_image]
    src = read_image(argv)
    cv.imshow("src", src)

    # [gray]
    gray = transform_grey(src)
    show_wait_destroy("gray", gray)


    # show_wait_overlay(src, cv.imread(gray))


    # [gray]
    # [bin]
    # Apply adaptiveThreshold at the bitwise_not of gray, notice the ~ symbol
    gray = cv.bitwise_not(gray)
    bw = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, \
                                cv.THRESH_BINARY, 15, -2)
    # Show binary image
    show_wait_destroy("binary", bw)

    # [bin]
    # [init]
    # Create the images that will use to extract the vertical lines
    vertical = np.copy(bw)

    # [vert]
    # Specify size on vertical axis
    rows = vertical.shape[0]
    verticalsize = rows // 30

    # Create structure element for extracting vertical lines through morphology operations
    verticalStructure = cv.getStructuringElement(cv.MORPH_RECT, (1, verticalsize))
    #show_wait_destroy("verticalStructure", verticalStructure)

    # Apply morphology operations
    vertical = cv.erode(vertical, verticalStructure)
    vertical = cv.dilate(vertical, verticalStructure, iterations=5)
    show_wait_destroy("vertical", vertical)
    vertical = cv.morphologyEx(vertical, cv.MORPH_CLOSE, verticalStructure)

    # Show extracted vertical lines
    show_wait_destroy("vertical", vertical)

    # [vert]
    # [smooth]
    # Inverse vertical image
    vertical = cv.bitwise_not(vertical)
    show_wait_destroy("vertical_bit", vertical)

    '''
    Extract edges and smooth image according to the logic
    1. extract edges
    2. dilate(edges)
    3. src.copyTo(smooth)
    4. blur smooth img
    5. smooth.copyTo(src, edges)
    '''
    # Step 1
    edges = cv.adaptiveThreshold(vertical, 255, cv.ADAPTIVE_THRESH_MEAN_C, \
                                cv.THRESH_BINARY, 3, -2)
    show_wait_destroy("edges", edges)

    # Step 2
    kernel = np.ones((2, 2), np.uint8)
    edges = cv.dilate(edges, kernel)
    show_wait_destroy("dilate", edges)

    # Step 3
    smooth = np.copy(vertical)

    # Step 4
    smooth = cv.blur(smooth, (2, 2))

    # Step 5
    (rows, cols) = np.where(edges != 0)
    vertical[rows, cols] = smooth[rows, cols]

    # Show final result
    show_wait_destroy("smooth - final", vertical)
    # [smooth]
    return 0

if __name__ == "__main__":
    main(sys.argv[1:])
