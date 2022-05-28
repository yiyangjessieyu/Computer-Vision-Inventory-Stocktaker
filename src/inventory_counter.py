# hough_line.py


# Imports needed for this program to run.
import cv2 as cv
import numpy as np

# Global image files paths to set
LOCAL_PATH = "/csse/users/yyu69/Desktop/COSC428/Project-april21/Computer-Vision-Inventory-Stocktaker/"
INPUT_IMAGE_PATH = 'resources/side_hearts.jpg'
OUTPUT_IMAGE_PATH = './resources/dark.png'

# Global hough normal settings to set.
CANNY_THRESHOLD2 = 200
HOUGH_THRESHOLD = 240


def display_hough_window():
    cv.namedWindow('Hough Line Transform')
    cv.createTrackbar('CannyThreshold1', 'Hough Line Transform', 0, 1200, nothing)
    cv.createTrackbar('CannyThreshold2', 'Hough Line Transform', CANNY_THRESHOLD2, 1200, nothing)
    cv.createTrackbar("HoughThreshold", 'Hough Line Transform', HOUGH_THRESHOLD, 1200, nothing)


def show_wait_destroy(window_name, img):
    cv.imshow(window_name, img)
    cv.moveWindow(window_name, 500, 0)
    cv.waitKey(0)
    cv.destroyWindow(window_name)


def nothing(x):
    # We need a callback for the createTrackbar function.
    # It doesn't need to do anything, however.
    pass


def get_tracker_values():
    canny_threshold1 = cv.getTrackbarPos('CannyThreshold1', 'Hough Line Transform')
    canny_threshold2 = cv.getTrackbarPos('CannyThreshold2', 'Hough Line Transform')
    hough_threshold = cv.getTrackbarPos('HoughThreshold', 'Hough Line Transform')
    return canny_threshold1, canny_threshold2, hough_threshold


def draw_hough_lines(lines):
    # Create a new copy of the original image for drawing line detections on.
    img = SOURCE_IMAGE.copy()

    # Create a dark image for drawing line detections on and later line refine.
    dark = np.zeros(img.shape)

    # Draw lines to be green
    if lines is not None:
        for line in lines:
            for rho, theta in line:
                if (theta >= np.radians(0) and theta <= np.radians(10)) or \
                    (theta >= np.radians(350) and theta <= np.radians(360)) or \
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


def extract_hough_normal():
    gray = cv.cvtColor(SOURCE_IMAGE, cv.COLOR_BGR2GRAY)
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


def transform_grey(src):
    # Transform source image to gray if it is not already
    if len(src.shape) != 2:
        return cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    else:
        return src


def transform_bitwise(src):
    # Apply adaptiveThreshold at the bitwise_not of gray, notice the ~ symbol
    src = cv.bitwise_not(src)
    bitwise = cv.adaptiveThreshold(src, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 15, -2)
    return bitwise


def extract_vertical(bitwise):
    # Create the images that will use to extract the vertical lines
    vertical = np.copy(bitwise)
    # Specify size on vertical axis
    rows = vertical.shape[0]
    vertical_size = rows // 30
    # Create structure element for extracting vertical lines through morphology operations
    verticalStructure = cv.getStructuringElement(cv.MORPH_RECT, (1, vertical_size))
    # show_wait_destroy("verticalStructure", verticalStructure)
    # Apply morphology operations
    vertical = cv.erode(vertical, verticalStructure)
    vertical = cv.dilate(vertical, verticalStructure, iterations=5)

    return vertical


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
        print ('Error opening image: ' + image_to_read)
        return -1

    # Scale the image down to 70% to fit on the monitor better.
    src = cv.resize(src, (int(src.shape[1] * 0.7), int(src.shape[0] * 0.7)))

    return src


def main():
    # [load_image]
    global SOURCE_IMAGE
    SOURCE_IMAGE = read_image(LOCAL_PATH + INPUT_IMAGE_PATH)
    cv.imshow("Source Image", SOURCE_IMAGE)

    # [hough lines]
    dark = extract_hough_normal()

    # [gray] TODO work on making this work as it cannot use this param type
    # TODO: if len(src.shape) != 2: AttributeError: 'NoneType' object has no attribute 'shape'
    # TODO: however using SOURCE_IMAGE as the input param works. so may be a type issue
    gray = transform_grey(dark)
    show_wait_destroy("gray", gray)

    # [bitwise]
    bitwise = transform_bitwise(gray)
    show_wait_destroy("bitwise", bitwise)


if __name__ == "__main__":
    main()
