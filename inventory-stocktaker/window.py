# window.py


# Imports needed for this program to run.
import cv2 as cv

# Global hough normal settings to set.
CANNY_THRESHOLD2 = 200
HOUGH_THRESHOLD = 240


def nothing(x):
    # We need a callback for the createTrackbar function.
    # It doesn't need to do anything, however.
    pass


def display_hough_window():
    cv.namedWindow('Hough Line Transform')
    cv.createTrackbar('CannyThreshold1', 'Hough Line Transform', 0, 1200, nothing)
    cv.createTrackbar('CannyThreshold2', 'Hough Line Transform', CANNY_THRESHOLD2, 1200, nothing)
    cv.createTrackbar("HoughThreshold", 'Hough Line Transform', HOUGH_THRESHOLD, 1200, nothing)  # Min Line Length


def get_tracker_values():
    canny_threshold1 = cv.getTrackbarPos('CannyThreshold1', 'Hough Line Transform')
    canny_threshold2 = cv.getTrackbarPos('CannyThreshold2', 'Hough Line Transform')
    hough_threshold = cv.getTrackbarPos('HoughThreshold', 'Hough Line Transform')  # Min Line Length
    return canny_threshold1, canny_threshold2, hough_threshold


def show_wait_destroy(window_name, img):
    cv.imshow(window_name, img)
    cv.moveWindow(window_name, 500, 0)
    cv.waitKey(0)
    cv.destroyWindow(window_name)
