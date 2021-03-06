# hough_line.py


# Imports needed for this program to run.
import cv2 as cv
import numpy as np
from window import *
from file import *


def draw_hough_lines(lines):
    # Create a new copy of the original image for drawing line detections on.
    img = SOURCE_IMAGE.copy()

    # Create a dark image for drawing line detections on and later line refine.
    dark = np.zeros(img.shape)

    # Draw lines to be green
    if lines is not None:
        for line in lines:
            for rho, theta in line:
                if (np.radians(0) <= theta <= np.radians(10)) or \
                    (np.radians(350) <= theta <= np.radians(360)) or \
                    (theta >= np.radians(170) and theta <= np.radians(190)):
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    x1 = int(x0 + 1000 * (-b))
                    y1 = int(y0 + 1000 * (a))
                    x2 = int(x0 - 1000 * (-b))
                    y2 = int(y0 - 1000 * (a))

                    cv.line(dark, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return img, dark


def extract_hough_normal(src):
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    display_hough_window()

    while True:
        # Get the updated tracker values from the hough window.
        canny_threshold1, canny_threshold2, hough_threshold = get_tracker_values()

        # Use the Canny Edge Detector to find some edges.
        edges = cv.Canny(gray, canny_threshold1, canny_threshold2)

        # Attempt to detect straight lines in the edge detected image.
        lines = cv.HoughLines(edges, 1, np.pi / 180, hough_threshold)

        # Draw each line that was detected.
        img, dark = draw_hough_lines(lines)

        # Create a combined image of Hough Line Transform result and the Canny Line Detector result.
        combined = np.concatenate((img, cv.cvtColor(edges, cv.COLOR_GRAY2BGR)), axis=1)

        # Show results.
        cv.imshow('Hough Line Transform', combined)
        cv.imshow('Dark', dark)

        # Save the dark drawing of lines onto desktop.
        cv.imwrite(LOCAL_PATH + OUTPUT_IMAGE_PATH, dark)

        # TODO erode then count lines again using hugh with length of the lines list

        if cv.waitKey(1000) & 0xFF == ord('q'):
            cv.destroyAllWindows()
            break

        dark = read_image(LOCAL_PATH + OUTPUT_IMAGE_PATH)
    return dark


def count_hough_normal(src):
    dst = cv.Canny(src, 50, 200, None, 3)
    edges = cv.Canny(src, 0, CANNY_THRESHOLD2)
    lines = cv.HoughLines(dst, 1, np.pi / 180, HOUGH_THRESHOLD)

    return len(lines)


def transform_grey(src):
    # Transform source image to gray if it is not already
    if len(src.shape) != 2:
        return cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    else:
        return src


def transform_bitwise(src):
    # Apply adaptiveThreshold at the bitwise_not of gray, notice the ~ symbol
    bitwise = cv.adaptiveThreshold(src, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 15, -2)
    return bitwise


def smooth_image(src):
    '''
        Extract edges and smooth image according to the logic
        1. extract edges
        2. dilate(edges)
        3. src.copyTo(smooth)
        4. blur smooth img
        5. smooth.copyTo(src, edges)
        '''
    # Step 1
    edges = cv.adaptiveThreshold(src, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 3, -2)
    show_wait_destroy("edges", edges)

    # Step 2
    kernel = np.ones((2, 2), np.uint8)
    edges = cv.dilate(edges, kernel)
    show_wait_destroy("dilate", edges)

    # Step 3
    smooth = np.copy(src)

    # Step 4
    smooth = cv.blur(smooth, (2, 2))

    # Step 5
    (rows, cols) = np.where(edges != 0)
    src[rows, cols] = smooth[rows, cols]

    return src


def main():
    results = {}

    # [load_image]
    global SOURCE_IMAGE
    SOURCE_IMAGE = read_image(LOCAL_PATH + INPUT_IMAGE_PATH)

    # [hough lines]
    dark = extract_hough_normal(SOURCE_IMAGE)

    # [gray]
    gray = transform_grey(dark)
    show_wait_destroy("gray", gray)

    # [bitwise]
    bitwise = transform_bitwise(gray)
    show_wait_destroy("bitwise", bitwise)

    kernel = np.ones((2, 2), np.uint8)
    dilate = cv.dilate(bitwise, kernel)
    show_wait_destroy("dilate", dilate)

    # [inverse bitwise]
    bitwise_inverse = cv.bitwise_not(bitwise)
    show_wait_destroy("bitwise_inverse", bitwise_inverse)

    # [smooth]
    smooth = smooth_image(bitwise_inverse)
    show_wait_destroy("smooth", smooth)

    results["dark"] = count_hough_normal(dark)
    results["gray"] = count_hough_normal(gray)
    results["bitwise"] = count_hough_normal(bitwise)
    results["dilate"] = count_hough_normal(dilate)
    results["bitwise_inverse"] = count_hough_normal(bitwise_inverse)
    results["smooth"] = count_hough_normal(smooth)

    output_results(results)


if __name__ == "__main__":
    main()
