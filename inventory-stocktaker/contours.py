# Imports needed for this program to run.
import cv2
import numpy as np


def extract_contours(image):
    # Create a dark image for drawing line detections on and later line refine.
    contour_dark = np.zeros(image.shape).astype("uint8")
    contour_dark2 = np.zeros(image.shape).astype("uint8")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Applying Gaussian blurring helps remove some of the high frequency edges
    # that we are not concerned with and allow us to obtain a more “clean” segmentation.
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # apply basic thresholding with parameter
    # 1. image to threshold,
    # 2. threshold to check;
    # if a pixel value is greater than our threshold we set it to be *black,
    # otherwise it is *white*
    (T, threshInv) = cv2.threshold(blurred, 200, 255,
                                   cv2.THRESH_BINARY_INV)
    cv2.imshow("Threshold Binary Inverse", threshInv)

    thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)[1]

    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts2 = cv2.findContours(threshInv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts2 = cnts2[0] if len(cnts2) == 2 else cnts2[1]


    for c in cnts:
        cv2.drawContours(image, [c], -1, (36, 255, 12), 3)
        cv2.drawContours(contour_dark, [c], -1, (36, 255, 12), 3)

    for c in cnts2:
        cv2.drawContours(contour_dark2, [c], -1, (36, 255, 12), 3)

    # visualize only the masked regions in the image
    masked = cv2.bitwise_and(image, image, mask=threshInv)
    cv2.imshow("Output", masked)
    cv2.waitKey(0)

    return thresh, image, contour_dark, contour_dark2
