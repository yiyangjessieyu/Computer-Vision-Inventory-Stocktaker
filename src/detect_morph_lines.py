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

def nothing(x):
    # We need a callback for the createTrackbar function.
    # It doesn't need to do anything, however.
    pass

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

def transform_bitwise(src):
    # Apply adaptiveThreshold at the bitwise_not of gray, notice the ~ symbol
    src = cv.bitwise_not(src)
    bitwise = cv.adaptiveThreshold(src, 255, cv.ADAPTIVE_THRESH_MEAN_C, \
                                cv.THRESH_BINARY, 15, -2)

    return bitwise

def extract_vertical(bitwise):
    # Create the images that will use to extract the vertical lines
    vertical = np.copy(bitwise)

    # Specify size on vertical axis
    rows = vertical.shape[0]
    vertical_size = rows // 30

    # Create structure element for extracting vertical lines through morphology operations
    verticalStructure = cv.getStructuringElement(cv.MORPH_RECT, (1, vertical_size))
    # show_wait_destroy("verticalStructure", verticalStructure)

    # Apply morphology operations
    vertical = cv.erode(vertical, verticalStructure)
    vertical = cv.dilate(vertical, verticalStructure, iterations=5)

    return vertical

def extract_hough_normal(src):
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

    #gray = convertToGrayBlur(img_original);

    #gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
    #gray = cv2.morphologyEx(gray, cv2.MORPH_DILATE, np.ones((7,7),np.uint8))

    cv.namedWindow('Hough Line Transform')
    cv.createTrackbar('CannyThreshold1', 'Hough Line Transform', 0, 1200, nothing)
    cv.createTrackbar('CannyThreshold2', 'Hough Line Transform', 200, 1200, nothing)
    cv.createTrackbar("HoughThreshold", 'Hough Line Transform', 240, 1200, nothing)

    while True:
        houghThreshold = cv.getTrackbarPos('HoughThreshold', 'Hough Line Transform')
        cannyThreshold1 = cv.getTrackbarPos('CannyThreshold1', 'Hough Line Transform')
        cannyThreshold2 = cv.getTrackbarPos('CannyThreshold2', 'Hough Line Transform')


        # Create a new copy of the original image for drawing on later.
        img = src.copy()
        # Use the Canny Edge Detector to find some edges.
        edges = cv.Canny(gray, cannyThreshold1, cannyThreshold2)
        # Attempt to detect straight lines in the edge detected image.
        lines = cv.HoughLines(edges, 1, np.pi/180, houghThreshold)

        dark = np.zeros(img.shape)

        # For each line that was detected, draw it on the img.
        if lines is not None:
            for line in lines:
                for rho,theta in line:
                    if (theta >= np.radians(0) and theta <= np.radians(10)) or \
                        (theta >= np.radians(350) and theta <= np.radians(360)) or \
                        (theta >= np.radians(170) and theta <= np.radians(190)) :
                        a = np.cos(theta)
                        b = np.sin(theta)
                        x0 = a*rho
                        y0 = b*rho
                        x1 = int(x0 + 1000*(-b))
                        y1 = int(y0 + 1000*(a))
                        x2 = int(x0 - 1000*(-b))
                        y2 = int(y0 - 1000*(a))

                        cv.line(dark,(x1,y1),(x2,y2),(0,255,0),2)
                        cv.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    else:
                        a = np.cos(theta)
                        b = np.sin(theta)
                        x0 = a*rho
                        y0 = b*rho
                        x1 = int(x0 + 1000*(-b))
                        y1 = int(y0 + 1000*(a))
                        x2 = int(x0 - 1000*(-b))
                        y2 = int(y0 - 1000*(a))

                        cv.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)


        # Create a combined image of Hough Line Transform result and the Canny Line Detector result.
        combined = np.concatenate((img, cv.cvtColor(edges, cv.COLOR_GRAY2BGR)), axis=1)

        cv.imshow('Hough Line Transform', combined)
        cv.imshow('Dark', dark)

        return dark

def main(argv):

    # [load_image]
    src = read_image(argv)
    cv.imshow("src", src)

    # [hough lines]
    dark = extract_hough_normal(src)

    # Apply morphology operations
    morph = cv.erode(dark, dark)
    morph = cv.dilate(dark, dark, iterations=5)
    cv.imshow("morph", morph)

    # [gray]
    #gray = transform_grey(dark)
    #show_wait_destroy("gray", gray)

    # [bitwise]
    #bitwise = transform_bitwise(dark)
    #show_wait_destroy("bitwise", bitwise)

    # [vertical]
    #vertical = extract_vertical(bitwise)
    #show_wait_destroy("vertical", vertical)

    # [inverse bitwise]
    #vertical_inverse = cv.bitwise_not(vertical)
    #show_wait_destroy("vertical_inverse", vertical_inverse)

    # [smooth]
    '''
    Extract edges and smooth image according to the logic
    1. extract edges
    2. dilate(edges)
    3. src.copyTo(smooth)
    4. blur smooth img
    5. smooth.copyTo(src, edges)
    '''
    # Step 1
    edges = cv.adaptiveThreshold(dark, 255, cv.ADAPTIVE_THRESH_MEAN_C, \
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
