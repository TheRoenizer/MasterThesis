# from https://www.geeksforgeeks.org/extract-images-from-video-in-python/
# Importing all necessary libraries
import cv2
import os

# Read the video from specified path
cam = cv2.VideoCapture("C:\\Users\\chris\\OneDrive - Syddansk Universitet\\SDU RobTek\\Master Thesis\\Segmentation_Robotic_Training\\Training\\Dataset1\\Video.avi")

try:

    # creating a folder named data
    if not os.path.exists('data'):
        os.makedirs('data')

    #if not created then raise error
except OSError:
    print('Error: Creating directory of data')

# frame
currentframe = 0

while (True):

    # reading from frame
    ret, frame = cam.read()

    if ret:
        # if video is still left continue creating images
        name = './data/frame' + str(currentframe) + '.png'
        print('Creating...' + name)

        # writing the extracted images
        cv2.imwrite(name, frame)

        # increasing counter so that it will
        # show how many frames are created
        currentframe += 1
    else:
        break

# Release all space and windows once done
cam.release()
cv2.destroyAllWindows()

