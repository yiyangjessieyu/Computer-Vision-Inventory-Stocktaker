# Inventory Stocker
### Author: Yiyang (Jessie) Yu, 2022
An application that can detect and count the number of common objects in a frame to aid in a store's stock-taking process. A tool like this can aid a store’s stock-taking process, as counting inventory can be a
redundant task that can be reduced. This file contains information that may be helpful to people who wish to extend this in future work.

## Application's Purpose
- Need a repetitive object detector for inventory on the shelf.
- Used by your regular worker, so need to be natural.
- Speed result from one input image, no time consumed by other model training.

## Setup
- Via Python 3.8
- OpenCV version: 4.5.5.64
- Device: Smartphone camera
- Test data: Photo from a smartphone camera

## Files of Interest
1. inventory_stocktaker.py: where the main code is contained.
2. file.py: where code for input/output processing is contained. Modify the local, input, or output global variables to your needs.
3. window.py: where code for displaying the process is contained. Modify to suit your window size needs.
4. test-data file: contains a few test data that can be used for this application.
5. Presensation.pdf: elevator pitch for the Inventory Stocktaker.

## For data analysis of result accuracy
- If you want to get an accuracy check:
1. name your input image {name}_{count}.jpg.
2. where {name} is a name given to your input image.
3. where {count} is the correct count of inventory in this image.
- If not, make sure the lines under # [data] in inventory.py is commented out.

## Running instructions
1. Get test data by taking photo images with your phone, or use one from the test-data file.
2. Make sure you have required setup.
3. Prepare your input image for accuracy analysis (or not).
4. Modify file.py, or window.py to suit your needs.
5. Run inventory_stocktaker.py.
6. click "q" to step through each process and proceed.

## Program process
1. The application will load the image. Code to refer to in file.py.
2. The contours in the image will be found and draw on a new dark background. Code to refer to in contour.py.
3. Filter out only the straight vertical lines from the contours on a dark background. Code to refer to in hough_line.py.
4. Dilate the vertical lines, therefore making them more pronounced and thicker. This also gets rid of multiple detections by merging close proximately lines together to be one.
5. Use hough line algorithm to count the vertical lines. Code to refer to in dilate.py.
6. Return count as stocktaking result. Code to refer to in hough_line.py.
