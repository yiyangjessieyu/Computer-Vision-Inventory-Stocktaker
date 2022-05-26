# hough_line.py

import cv2 as cv
import numpy as np

LOCAL_PATH = "/csse/users/yyu69/Desktop/COSC428/Project-april21/Computer-Vision-Inventory-Stocktaker/"
INPUT_IMAGE_PATH = 'resources/side_tape.jpg'
OUTPUT_IMAGE_PATH = './resources/dark.png'

def nothing(x):
    # We need a callback for the createTrackbar function.
    # It doesn't need to do anything, however.
    pass

def extract_hough_normal(src):

    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

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
                    # else:
                    #     a = np.cos(theta)
                    #     b = np.sin(theta)
                    #     x0 = a*rho
                    #     y0 = b*rho
                    #     x1 = int(x0 + 1000*(-b))
                    #     y1 = int(y0 + 1000*(a))
                    #     x2 = int(x0 - 1000*(-b))
                    #     y2 = int(y0 - 1000*(a))
                    #
                    #     cv.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)


        # Create a combined image of Hough Line Transform result and the Canny Line Detector result.
        combined = np.concatenate((img, cv.cvtColor(edges, cv.COLOR_GRAY2BGR)), axis=1)

        cv.imshow('Hough Line Transform', combined)
        cv.imshow('Dark', dark)
        cv.imwrite(LOCAL_PATH + OUTPUT_IMAGE_PATH, dark)

        # erode
        # then count lines again using hugh with length of the lines list

        if cv.waitKey(1000) & 0xFF == ord('q'):
            cv.destroyAllWindows()
            break


def read_image(image_to_read):
    # Check number of arguments
    if len(image_to_read) < 1:
        print ('Not enough parameters')
        print ('Usage:\nmorph_lines_detection.py < path_to_image >')
        return -1

    # Load the image
    src = cv.imread(image_to_read, cv.IMREAD_COLOR)

    # Check if image is loaded fine
    if src is None:
        print ('Error opening image: ' + argv[0])
        return -1

    # Scale the image down to 70% to fit on the monitor better.
    src = cv.resize(src, (int(src.shape[1] * 0.7), int(src.shape[0] * 0.7)))

    return src

def main():
    # [load_image]
    src = read_image(LOCAL_PATH + INPUT_IMAGE_PATH)
    cv.imshow("src", src)

    # [hough lines]
    dark = extract_hough_normal(src)

if __name__ == "__main__":
    main()
