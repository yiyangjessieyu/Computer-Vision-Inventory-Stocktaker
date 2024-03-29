# Imports needed for this program to run.
import cv2 as cv

# Global input file paths to set.
LOCAL_PATH = "/csse/users/yyu69/Desktop/COSC428/Project-april21/Computer-Vision-Inventory-Stocktaker/inventory-stocktaker/test-data/"
INPUT_IMAGE_PATH = 'button_battery_24.jpg'

# Global output file paths to set.
OUTPUT_IMAGE_PATH = 'dark.png'
OUTPUT_FILE_PATH = 'output.txt'

# Global window size to set
WINDOW_RATIO = 0.2


def save_image(image_to_save):
    cv.imwrite('saved_image.jpg', image_to_save)


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

    # Scale the image down to fit on the monitor better.
    src = cv.resize(src, (int(src.shape[1] * WINDOW_RATIO), int(src.shape[0] * WINDOW_RATIO)))

    return src


def output_results(results, contour_method, hough_method):
    # Opening a file
    output = open(LOCAL_PATH + OUTPUT_FILE_PATH, 'a')

    # Writing a string to file
    for key, value in results.items():
        correct_count, result_count = value

        if correct_count == 0 or result_count == 0:
            accuracy = 0
        else:
            accuracy = float(correct_count)/float(result_count)

        output.write("For the image: " + str(key) + '\n'
                     "With contour method: " + str(contour_method) + '\n'
                     "With hough method: " + str(hough_method) + '\n'
                     "Correct count is: " + str(correct_count) + '\n'
                     "Result count is: " + str(result_count) + '\n'
                     "Accuracy of: " + str(accuracy) + '\n'
                     "-----------------------------------------------" + '\n')

    # Closing file
    output.close()

    # Checking if the data is written to file or not
    output = open(LOCAL_PATH + OUTPUT_FILE_PATH, 'r')
    print(output.read())
    output.close()
