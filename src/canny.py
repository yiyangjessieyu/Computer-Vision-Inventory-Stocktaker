# canny.py

import cv2
import numpy as np

LOCAL_PATH = "/csse/users/yyu69/Desktop/COSC428/Project-april21/Computer-Vision-Inventory-Stocktaker/"

def nothing(x):
    # We need a callback for the createTrackbar function.
    # It doesn't need to do anything, however.
    pass


img = cv2.imread(LOCAL_PATH + 'resources/input_image2.jpg', 0)
# Scale the image down to 70% to fit on the monitor better.
img = cv2.resize(img, (int(img.shape[1]*0.7), int(img.shape[0]*0.7)))

# Create a window and add two trackbars for controlling the thresholds.
cv2.namedWindow('Canny Edge Detection')
cv2.createTrackbar('Threshold1', 'Canny Edge Detection', 0, 1200, nothing)
cv2.createTrackbar('Threshold2', 'Canny Edge Detection', 0, 1200, nothing)

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
