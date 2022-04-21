LOCAL_PATH = "/csse/users/yyu69/Desktop/COSC428/Project-april21/Computer-Vision-Inventory-Stocktaker/"
INPUT_IMAGE_PATH = 'resources/input_image2.jpg'
TEMPLATE_IMAGE_PATH = 'resources/template_image2.png'

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# Importing input images
input_image = cv.imread(LOCAL_PATH + INPUT_IMAGE_PATH)
input_image_gray = cv.cvtColor(input_image, cv.COLOR_BGR2GRAY)
input_image_copy = input_image.copy()

# Importing template images
template_image = cv.imread(LOCAL_PATH + TEMPLATE_IMAGE_PATH,0)
template_width, template_height = template_image.shape[::-1]

# All the 6 methods for comparison in a list
methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
            'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']

for method in methods:

    # Set up
    img = input_image_copy.copy()
    current_method = eval(method)

    # Thresholding for when there's multiple template image instances in the input image to search for
    result = cv.matchTemplate(input_image_gray, template_image, cv.TM_CCOEFF_NORMED)
    threshold = 0.48
    loc = np.where(result >= threshold)
    for pt in zip(*loc[::-1]):
      cv.rectangle(input_image, pt, (pt[0] + template_width, pt[1] + template_height), (0, 0, 255), 2)
    cv.imwrite('result.png', input_image)

    # cv.minMaxLoc function for when there's one template image instance in the input image to search for
    # If the above is true, then below code must be commented out
    # [ONE TEMPLATE IMAGE INSTANCE CODE STARTING HERE]
    #
    # # Apply template Matching
    # result = cv.matchTemplate(img, template_image, current_method)
    #
    # min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
    #
    # # If the current_method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    # if current_method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
    #     top_left = min_loc
    # else:
    #     top_left = max_loc
    #
    # bottom_right = (top_left[0] + template_width, top_left[1] + template_height)
    # cv.rectangle(img,top_left, bottom_right, 255, 2)
    # plt.subplot(121),plt.imshow(result,cmap = 'gray')
    # plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    # plt.subplot(122),plt.imshow(img,cmap = 'gray')
    # plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    # plt.suptitle(method)
    # plt.show()
    # [ONE TEMPLATE IMAGE INSTANCE CODE ENDING HERE]

