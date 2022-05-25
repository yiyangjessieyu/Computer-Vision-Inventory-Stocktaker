# canny.py

import cv2
import numpy as np

LOCAL_PATH = "/csse/users/yyu69/Desktop/COSC428/Project-april21/Computer-Vision-Inventory-Stocktaker/"
INPUT_IMAGE_PATH = "resources/side_crayons.jpg"

def nothing(x):
    # We need a callback for the createTrackbar function.
    # It doesn't need to do anything, however.
    pass


img = cv2.imread(LOCAL_PATH + INPUT_IMAGE_PATH, 0)
# Scale the image down to 70% to fit on the monitor better.
img = cv2.resize(img, (int(img.shape[1]*0.7), int(img.shape[0]*0.7)))

# Create a window and add two trackbars for controlling the thresholds.
cv2.namedWindow('Canny Edge Detection')
cv2.createTrackbar('Threshold1', 'Canny Edge Detection', 0, 1200, nothing)
cv2.createTrackbar('Threshold2', 'Canny Edge Detection', 0, 1200, nothing)

#TODO https://towardsdatascience.com/canny-edge-detection-step-by-step-in-python-computer-vision-b49c3a2d8123
def gaussian_kernel(size, sigma=1):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g

while True:
    # Get the latest threshold values.
    threshold1 = cv2.getTrackbarPos('Threshold1', 'Canny Edge Detection')
    threshold2 = cv2.getTrackbarPos('Threshold2', 'Canny Edge Detection')

    # Update the image using the latest threshold.
    edges = cv2.Canny(img, threshold1, threshold2)
    cv2.imshow('Canny Edge Detection', edges)

    if cv2.waitKey(100) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
