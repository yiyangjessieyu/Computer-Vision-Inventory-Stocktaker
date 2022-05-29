# hough_line.py

import cv2
import numpy as np

# Global threshold settings
CANNY_THRESHOLD_1 = 1200
CANNY_THRESHOLD_2 = 0
HOUGH_THRESHOLD_1 = 120


def nothing(x):
    # We need a callback for the createTrackbar function.
    # It doesn't need to do anything, however.
    pass


def count_hough_lines(img_original):
    gray = img_original
    # Use the Canny Edge Detector to find some edges.
    edges = cv2.Canny(gray, CANNY_THRESHOLD_1, CANNY_THRESHOLD_2)
    # Attempt to detect straight lines in the edge detected image.
    lines = cv2.HoughLines(edges, 1, np.pi / 180, HOUGH_THRESHOLD_1)

    return len(lines)
