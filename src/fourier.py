from __future__ import print_function
import sys
import cv2 as cv
import numpy as np

# Global input file paths to set.
LOCAL_PATH = "/csse/users/yyu69/Desktop/COSC428/Project-april21/Computer-Vision-Inventory-Stocktaker/resources/"
INPUT_IMAGE_PATH = 'side_hearts_9.jpg'

WINDOW_RATIO = 0.4

def print_help():
    print('''
    This program demonstrated the use of the discrete Fourier transform (DFT).
    The dft of an image is taken and it's power spectrum is displayed.
    Usage:
    discrete_fourier_transform.py [image_name -- default lena.jpg]''')


def main():
    print_help()

    I = cv.imread("dilate.png", cv.IMREAD_GRAYSCALE)
    if I is None:
        print('Error opening image')
        return -1

    I = cv.resize(I, (int(I.shape[1] * WINDOW_RATIO), int(I.shape[0] * WINDOW_RATIO)))

    # Expand the image to an optimal size
    rows, cols = I.shape
    m = cv.getOptimalDFTSize(rows)
    n = cv.getOptimalDFTSize(cols)
    padded = cv.copyMakeBorder(I, 0, m - rows, 0, n - cols, cv.BORDER_CONSTANT, value=[0, 0, 0])

    # Make place for both the complex and the real values
    planes = [np.float32(padded), np.zeros(padded.shape, np.float32)]
    complexI = cv.merge(planes)  # Add to the expanded another plane with zeros
    cv.dft(complexI, complexI)  # this way the result may fit in the source matrix

    # Transform the real and complex values to magnitude
    cv.split(complexI, planes)  # planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    cv.magnitude(planes[0], planes[1], planes[0])  # planes[0] = magnitude
    magI = planes[0]

    # Switch to a logarithmic scale
    matOfOnes = np.ones(magI.shape, dtype=magI.dtype)
    cv.add(matOfOnes, magI, magI)  # switch to logarithmic scale
    cv.log(magI, magI)

    # Prepare to crop and rearrange
    magI_rows, magI_cols = magI.shape

    # Crop the spectrum, if it has an odd number of rows or columns
    magI = magI[0:(magI_rows & -2), 0:(magI_cols & -2)]
    cx = int(magI_rows / 2)
    cy = int(magI_cols / 2)
    q0 = magI[0:cx, 0:cy]  # Top-Left - Create a ROI per quadrant
    q1 = magI[cx:cx + cx, 0:cy]  # Top-Right
    q2 = magI[0:cx, cy:cy + cy]  # Bottom-Left
    q3 = magI[cx:cx + cx, cy:cy + cy]  # Bottom-Right
    tmp = np.copy(q0)  # swap quadrants (Top-Left with Bottom-Right)
    magI[0:cx, 0:cy] = q3
    magI[cx:cx + cx, cy:cy + cy] = tmp
    tmp = np.copy(q1)  # swap quadrant (Top-Right with Bottom-Left)
    magI[cx:cx + cx, 0:cy] = q2
    magI[0:cx, cy:cy + cy] = tmp

    # Normalize
    cv.normalize(magI, magI, 0, 1, cv.NORM_MINMAX)  # Transform the matrix with float values into a

    # Window details
    cv.imshow("Input Image", I)  # Show the result
    cv.imshow("spectrum magnitude", magI)
    cv.waitKey()


if __name__ == "__main__":
    main()
