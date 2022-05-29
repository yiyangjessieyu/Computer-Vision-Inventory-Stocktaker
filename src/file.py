# Imports needed for this program to run.
import cv2 as cv

# Global input file paths to set.
LOCAL_PATH = "/csse/users/yyu69/Desktop/COSC428/Project-april21/Computer-Vision-Inventory-Stocktaker/resources/"
INPUT_IMAGE_PATH = 'side_hearts2'

# Global output file paths to set.
OUTPUT_FILE_PATH = 'src/output.txt'

# Global window size to set
WINDOW_RATIO = 0.6


def read_image(image_to_read):
    # Check number of arguments
    if len(image_to_read) < 1:
        print ('Not enough parameters')
        print ('Usage:\nmorph_lines_detection.py < path_to_image >')
        return -1

    # Load the image
    src = cv.imread(image_to_read + ".jpg", cv.IMREAD_COLOR)

    # Check if image is loaded fine
    if src is None:
        print ('Error opening image: ' + image_to_read)
        return -1

    # Scale the image down to 70% to fit on the monitor better.
    src = cv.resize(src, (int(src.shape[1] * WINDOW_RATIO), int(src.shape[0] * WINDOW_RATIO)))

    return src


def save_image(image, file_name):
    print(LOCAL_PATH + str(file_name))
    cv.imwrite(LOCAL_PATH + str(file_name) + ".jpg", image)


def output_results(results):
    # Opening a file
    output = open(LOCAL_PATH + OUTPUT_FILE_PATH, 'w')

    # Writing a string to file
    for key, value in results.items():
        output.write(str(key) + ' ' + str(value) + '\n')

    # Closing file
    output.close()

    # Checking if the data is written to file or not
    output = open(LOCAL_PATH + OUTPUT_FILE_PATH, 'r')
    print(output.read())
    output.close()
