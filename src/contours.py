# Imports needed for this program to run.
import cv2
import numpy as np
from file import *


def extract_contours(image):
    # Create a dark image for drawing line detections on and later line refine.
    contour_dark = np.zeros(image.shape)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)[1]

    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    for c in cnts:
        cv2.drawContours(image, [c], -1, (36, 255, 12), 3)
        cv2.drawContours(contour_dark, [c], -1, (36, 255, 12), 3)

    save_image(contour_dark, "contour_dark")

    return thresh, image, contour_dark
