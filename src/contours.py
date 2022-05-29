# Imports needed for this program to run.
import cv2
import numpy as np

# Global input file paths to set.
LOCAL_PATH = "/csse/users/yyu69/Desktop/COSC428/Project-april21/Computer-Vision-Inventory-Stocktaker/"
INPUT_IMAGE_PATH = 'resources/side_tape.jpg'

image = cv2.imread(LOCAL_PATH + INPUT_IMAGE_PATH)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)[1]

cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

lines = 0
for c in cnts:
    cv2.drawContours(image, [c], -1, (36, 255, 12), 3)
    lines += 1

print(lines)
cv2.imshow('thresh', thresh)
cv2.imshow('image', image)
cv2.waitKey()
