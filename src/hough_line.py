# hough_line.py

import cv2
import numpy as np
from file import *

# Global threshold settings
CANNY_THRESHOLD_1 = 1200
CANNY_THRESHOLD_2 = 0
HOUGH_THRESHOLD_1 = 120

def nothing(x):
    # We need a callback for the createTrackbar function.
    # It doesn't need to do anything, however.
    pass


def houghNormal(img_original):
    gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)

    cv2.namedWindow('Hough Line Transform')
    cv2.createTrackbar('CannyThreshold1', 'Hough Line Transform', CANNY_THRESHOLD_1, 1200, nothing)
    cv2.createTrackbar('CannyThreshold2', 'Hough Line Transform', CANNY_THRESHOLD_2, 1200, nothing)
    cv2.createTrackbar("HoughThreshold", 'Hough Line Transform', HOUGH_THRESHOLD_1, 1200, nothing)

    while True:
        houghThreshold = cv2.getTrackbarPos('HoughThreshold', 'Hough Line Transform')
        cannyThreshold1 = cv2.getTrackbarPos('CannyThreshold1', 'Hough Line Transform')
        cannyThreshold2 = cv2.getTrackbarPos('CannyThreshold2', 'Hough Line Transform')

        # Create a new copy of the original image for drawing on later.
        img = img_original.copy()
        # Create a dark image for drawing line detections on and later line refine.
        dark = np.zeros(img.shape)
        # Use the Canny Edge Detector to find some edges.
        edges = cv2.Canny(gray, cannyThreshold1, cannyThreshold2)
        # Attempt to detect straight lines in the edge detected image.
        lines = cv2.HoughLines(edges, 1, np.pi / 180, houghThreshold)

        # For each line that was detected, draw it on the img.
        if lines is not None:
            for line in lines:
                for rho, theta in line:
                    if (np.radians(0) <= theta <= np.radians(2)) or \
                        (np.radians(358) <= theta <= np.radians(360)) or \
                        (theta >= np.radians(178) and theta <= np.radians(182)):
                        a = np.cos(theta)
                        b = np.sin(theta)
                        x0 = a * rho
                        y0 = b * rho
                        x1 = int(x0 + 1000 * (-b))
                        y1 = int(y0 + 1000 * (a))
                        x2 = int(x0 - 1000 * (-b))
                        y2 = int(y0 - 1000 * (a))
                        cv2.line(dark, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Create a combined image of Hough Line Transform result and the Canny Line Detector result.
        combined = np.concatenate((img, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)), axis=1)

        cv2.imshow('Hough Line Transform', combined)

        if cv2.waitKey(1000) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

    return dark

def houghP(img_original):
    img_original = cv2.imread('images/gundam.jpg')

    blur = cv2.GaussianBlur(img_original, (9, 9), 0)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

    cv2.namedWindow('Hough Line Transform')
    cv2.createTrackbar('Canny Threshold 1', 'Hough Line Transform', 0, 1200, nothing)
    cv2.createTrackbar('Canny Threshold 2', 'Hough Line Transform', 0, 1200, nothing)
    cv2.createTrackbar("Min Line Length", 'Hough Line Transform', 0, 100, nothing)
    cv2.createTrackbar("Max Line Gap", 'Hough Line Transform', 0, 100, nothing)

    while True:
        minLineLength = cv2.getTrackbarPos('Min Line Length', 'Hough Line Transform')
        maxLineGap = cv2.getTrackbarPos('Max Line Gap', 'Hough Line Transform')
        cannyThreshold1 = cv2.getTrackbarPos('Canny Threshold 1', 'Hough Line Transform')
        cannyThreshold2 = cv2.getTrackbarPos('Canny Threshold 2', 'Hough Line Transform')

        # Create a new copy of the original image for drawing on later.
        img = img_original.copy()
        # Create a dark image for drawing line detections on and later line refine.
        dark = np.zeros(img.shape)
        # Use the Canny Edge Detector to find some edges.
        edges = cv2.Canny(gray, cannyThreshold1, cannyThreshold2)
        # Attempt to detect straight lines in the edge detected image.
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=minLineLength, maxLineGap=maxLineGap)

        # For each line that was detected, draw it on the img.
        if lines is not None:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(dark, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Create a combined image of Hough Line Transform result and the Canny Line Detector result.
        combined = np.concatenate((img, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)), axis=1)

        cv2.imshow('Hough Line Transform', combined)

        if cv2.waitKey(100) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

    return dark
