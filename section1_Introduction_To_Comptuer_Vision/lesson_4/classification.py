import cv2 # computer vision library
import helpers

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def avg_brightness(rgb_image):
    # Convert image to HSV
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    # Add up all the pixel values in the V channel
    sum_brightness = np.sum(hsv[:,:,2])
    area = 600*1100.0  # pixels
    # find the avg
    avg = sum_brightness/area
    return avg

# This function should take in RGB image input
def estimate_label(rgb_image):
    
    ## TODO: extract average brightness feature from an RGB image 
    a_b = avg_brightness(rgb_image)
    # Use the avg brightness feature to predict a label (0, 1)
    predicted_label = 0
    
    ## TODO: set the value of a threshold that will separate day and night images
    # print("estimated bright =",a_b)
    if a_b > 103:
        predicted_label= 1
    ## TODO: Return the predicted_label (0 or 1) based on whether the avg is 
    # above or below the threshold
    
    return predicted_label    
    


# Image data directories
image_dir_training = "day_night_images/training/"
image_dir_test = "day_night_images/test/"
# Using the load_dataset function in helpers.py
# Load training data
IMAGE_LIST = helpers.load_dataset(image_dir_training)

STANDARDIZED_LIST = helpers.standardize(IMAGE_LIST)

# Select an image by index
# image_num = 0
# selected_image = STANDARDIZED_LIST[image_num][0]
# selected_label = STANDARDIZED_LIST[image_num][1]

# Display image and data about it
# plt.imshow(selected_image)
# print("Shape: "+str(selected_image.shape))
# print("Label [1 = day, 0 = night]: " + str(selected_label))

# As an example, a "night" image is loaded in and its avg brightness is displayed
# image_num = 190
# test_im = STANDARDIZED_LIST[image_num][0]
total_img = len(STANDARDIZED_LIST)
correct = 0
for (img, actual_lbl) in STANDARDIZED_LIST:
    # (img, actual_label) = STANDARDIZED_LIST[0]
    avg = avg_brightness(img)
    # print('Avg brightness: ' + str(avg))

    predicted_lbl = estimate_label(img)
    if predicted_lbl == actual_lbl:
        correct+=1
    else :
        print("Predicted", predicted_lbl)
        print("Actual", actual_lbl)

print("Accuracy = ", correct*100/total_img)