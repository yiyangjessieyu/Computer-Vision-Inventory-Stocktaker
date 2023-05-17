# Imports needed for this program to run.
import cv2
import numpy as np
import numpy as np
import matplotlib.pyplot as plt

def extract_contours(image, contour_method):
    # Create a dark image for drawing line detections on and later line refine.
    contour_dark = np.zeros(image.shape).astype("uint8")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Applying Gaussian blurring helps remove some of the high frequency edges
    # that we are not concerned with and allow us to obtain a more “clean” segmentation.
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    cv2.imshow("blurred", blurred)

    # apply thresholding with parameter
    # 1. image to threshold,
    # 2. threshold to check;
    # if a pixel value is greater than our threshold we set it to be *black,
    # otherwise it is *white*

    # instead of manually specifying the threshold value, we can use
    # adaptive thresholding to examine neighborhoods of pixels and
    # adaptively threshold each neighborhood

    if contour_method == "NORMAL":
        threshInv = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)[1]
        cv2.imshow("NORMAL threshInv", threshInv)

    elif contour_method == "OTSU":
        (T, threshInv) = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        cv2.imshow("OTSU threshInv", threshInv)

    elif contour_method == "ADAPT":
        threshInv = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10)
        cv2.imshow("Mean Adaptive threshInv", threshInv)

    elif contour_method == "ALL":
        basic = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)[1]
        cv2.imshow("Basic", basic)

        (T, otsu) = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        cv2.imshow("Otsu", otsu)

        adaptive = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10)
        cv2.imshow("Mean Adaptive", adaptive)

        # visualize only the masked regions in the image
        m_basic = cv2.bitwise_and(image, image, mask=basic)
        cv2.imshow("masked m_basic", m_basic)

        m_otsu = cv2.bitwise_and(image, image, mask=otsu)
        cv2.imshow("masked m_otsu", m_otsu)

        m_adaptive = cv2.bitwise_and(image, image, mask=adaptive)
        cv2.imshow("masked m_adaptive", m_adaptive)

    cv2.waitKey(0)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # dilate = cv2.dilate(threshInv, kernel, iterations=1)
    # cv2.imshow("dilate Mean Adaptive Thresholding", dilate)
    # cv2.waitKey(0)

    # close = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, kernel, iterations=2)
    # cv2.imshow(" close Mean Adaptive Thresholding", close)
    # cv2.waitKey(0)

    # visualize only the masked regions in the image

    if contour_method != "ALL":
        masked = cv2.bitwise_and(image, image, mask=threshInv)
        cv2.imshow("masked threshInv", masked)
        cv2.waitKey(0)

        cnts = cv2.findContours(threshInv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        print("product count in the image from contours : ", len(cnts))

        for c in cnts:
            cv2.drawContours(image, [c], -1, (36, 255, 12), 2)
            cv2.drawContours(contour_dark, [c], -1, (36, 255, 12), 2)

        cv2.imshow("image", image)
        cv2.waitKey(0)


    # (cnt, hierarchy) = cv2.findContours(
    #     threshInv.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # cv2.drawContours(rgb, cnt, -1, (0, 255, 0), 2)
    #
    # plt.imshow(rgb)

    thresholds = [basic, otsu, adaptive]
    contours = []

    for threshInv in thresholds:
        cnts = cv2.findContours(threshInv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        print("product count in the image from contours : ", len(cnts))

        for c in cnts:
            cv2.drawContours(image, [c], -1, (36, 255, 12), 2)
            cv2.drawContours(contour_dark, [c], -1, (36, 255, 12), 2)

        contours.append((image, contour_dark))

    cv2.imshow("image", contours[0])
    cv2.imshow("image", contours[1])
    cv2.imshow("image", contours[2])

    return contour_dark
