import cv2 # computer vision library
import helpers
import random

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Image data directories
image_dir_training = "day_night_images/training/"
image_dir_test = "day_night_images/test/"
# Using the load_dataset function in helpers.py
# Load training data
IMAGE_LIST = helpers.load_dataset(image_dir_training)
# Standardize all training images
STANDARDIZED_LIST = helpers.standardize(IMAGE_LIST)

# Find the average Value or brightness of an image
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
    # TO-DO: Extract average brightness feature from an RGB image 
    avg = avg_brightness(rgb_image)
    # Use the avg brightness feature to predict a label (0, 1)
    predicted_label = 0
    # TO-DO: Try out different threshold values to see what works best!
    threshold = 102
    if(avg > threshold):
        # if the average brightness is above the threshold value, we classify it as "day"
        predicted_label = 1
    # else, the predicted_label can stay 0 (it is predicted to be "night")
    return predicted_label    


# Constructs a list of misclassified images given a list of test images and their labels
def get_misclassified_images(test_images):
    # Track misclassified images by placing them into a list
    misclassified_images_labels = []

    # Iterate through all the test images
    # Classify each image and compare to the true label
    for image in test_images:
        # Get true data
        im = image[0]
        true_label = image[1]
        # Get predicted label from your classifier
        predicted_label = estimate_label(im)
        # Compare true and predicted labels 
        if(predicted_label != true_label):
            # If these labels are not equal, the image has been misclassified
            misclassified_images_labels.append((im, predicted_label, true_label))
    # Return the list of misclassified [image, predicted_label, true_label] values
    return misclassified_images_labels


# Using the load_dataset function in helpers.py
# Load test data
TEST_IMAGE_LIST = helpers.load_dataset(image_dir_test)
# Standardize the test data
STANDARDIZED_TEST_LIST = helpers.standardize(TEST_IMAGE_LIST)
# Shuffle the standardized test data
random.shuffle(STANDARDIZED_TEST_LIST)


MISCLASSIFIED = get_misclassified_images(STANDARDIZED_TEST_LIST)
# Accuracy calculations
total = len(STANDARDIZED_TEST_LIST)
num_correct = total - len(MISCLASSIFIED)
accuracy = num_correct/total
print('Accuracy: ' + str(accuracy))
print("Number of misclassified images = " + str(len(MISCLASSIFIED)) +' out of '+ str(total))



# Visualize misclassified example(s)
num = 0
test_mis_im = MISCLASSIFIED[num][0]

## TODO: Display an image in the `MISCLASSIFIED` list 
fig = plt.figure(figsize = (16,16))
plt.title("Misclassified images")
for index in range(len(MISCLASSIFIED)):
    ax = fig.add_subplot(4,4,index+1, xticks=[],yticks=[])
    image = MISCLASSIFIED[index][0]
    label_true = MISCLASSIFIED[index][1]
    label_guess = MISCLASSIFIED[index][2]
    bright = avg_brightness(image)
    ax.imshow(image)
    ax.set_title("{}  {}  {} {}    {:0.0f}    {}".format('Truth', 'Brightness', 'Prediction\n', label_true, bright, label_guess))
    if index==15:
        break
plt.show()
## TODO: Print out its predicted label - 
## to see what the image *was* incorrectly classified as