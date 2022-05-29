# Imports needed for this program to run.
import cv2
import numpy as np


def extract_contours(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)[1]

    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    lines = 0
    for c in cnts:
        cv2.drawContours(image, [c], -1, (36, 255, 12), 3)
        lines += 1

    print(lines)

    return thresh, image
