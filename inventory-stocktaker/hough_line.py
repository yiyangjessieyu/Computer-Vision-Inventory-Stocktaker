# hough_line.py

import cv2
import numpy as np

HOUGH_THRESHOLD = 125

def nothing(x):
    # We need a callback for the createTrackbar function.
    # It doesn't need to do anything, however.
    pass

def count_houghNormal(img_original):
    #gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
    gray = img_original
    cv2.namedWindow('Hough Line Transform')
    cv2.createTrackbar('CannyThreshold1', 'Hough Line Transform', 0, 1200, nothing)
    cv2.createTrackbar('CannyThreshold2', 'Hough Line Transform', 0, 1200, nothing)
    cv2.createTrackbar("HoughThreshold", 'Hough Line Transform', HOUGH_THRESHOLD, 1200, nothing)

    while True:
        houghThreshold = cv2.getTrackbarPos('HoughThreshold', 'Hough Line Transform')
        cannyThreshold1 = cv2.getTrackbarPos('CannyThreshold1', 'Hough Line Transform')
        cannyThreshold2 = cv2.getTrackbarPos('CannyThreshold2', 'Hough Line Transform')

        # Create a new copy of the original image for drawing on later.
        img = img_original.copy()
        # Create a dark image for drawing line detections on and later line refine.
        hough_dark = np.zeros(img_original.shape).astype("uint8")
        # Use the Canny Edge Detector to find some edges.
        edges = cv2.Canny(gray, cannyThreshold1, cannyThreshold2)
        # Attempt to detect straight lines in the edge detected image.
        lines = cv2.HoughLines(edges, 1, np.pi / 180, houghThreshold)

        count = 0

        #For each line that was detected, draw it on the img.
        if lines is not None:
            for line in lines:
                for rho, theta in line:
                    if (np.radians(0) <= theta <= np.radians(1)) or \
                        (np.radians(359) <= theta <= np.radians(360)) or \
                        (theta >= np.radians(179) and theta <= np.radians(181)):
                        a = np.cos(theta)
                        b = np.sin(theta)
                        x0 = a * rho
                        y0 = b * rho
                        x1 = int(x0 + 1000 * (-b))
                        y1 = int(y0 + 1000 * (a))
                        x2 = int(x0 - 1000 * (-b))
                        y2 = int(y0 - 1000 * (a))
                        cv2.line(hough_dark, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        count += 1

        if cv2.waitKey(1000) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

    return count


def houghNormal(img_original):
    gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)

    cv2.namedWindow('Hough Line Transform')
    cv2.createTrackbar('CannyThreshold1', 'Hough Line Transform', 0, 1200, nothing)
    cv2.createTrackbar('CannyThreshold2', 'Hough Line Transform', 0, 1200, nothing)
    cv2.createTrackbar("HoughThreshold", 'Hough Line Transform', 0, 1200, nothing)

    while True:
        houghThreshold = cv2.getTrackbarPos('HoughThreshold', 'Hough Line Transform')
        cannyThreshold1 = cv2.getTrackbarPos('CannyThreshold1', 'Hough Line Transform')
        cannyThreshold2 = cv2.getTrackbarPos('CannyThreshold2', 'Hough Line Transform')

        # Create a new copy of the original image for drawing on later.
        img = img_original.copy()
        # Create a dark image for drawing line detections on and later line refine.
        hough_dark = np.zeros(img_original.shape).astype("uint8")
        # Use the Canny Edge Detector to find some edges.
        edges = cv2.Canny(gray, cannyThreshold1, cannyThreshold2)
        # Attempt to detect straight lines in the edge detected image.
        lines = cv2.HoughLines(edges, 1, np.pi / 180, houghThreshold)

        # For each line that was detected, draw it on the img.
        if lines is not None:
            for line in lines:
                for rho, theta in line:
                    if (np.radians(0) <= theta <= np.radians(4)) or \
                        (np.radians(356) <= theta <= np.radians(360)) or \
                        (theta >= np.radians(176) and theta <= np.radians(184)):
                        a = np.cos(theta)
                        b = np.sin(theta)
                        x0 = a * rho
                        y0 = b * rho
                        x1 = int(x0 + 1000 * (-b))
                        y1 = int(y0 + 1000 * (a))
                        x2 = int(x0 - 1000 * (-b))
                        y2 = int(y0 - 1000 * (a))
                        cv2.line(hough_dark, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Create a combined image of Hough Line Transform result and the Canny Line Detector result.
        combined = np.concatenate((img, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)), axis=1)

        cv2.imshow('Hough Line Transform', combined)

        if cv2.waitKey(1000) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

    return img, hough_dark
