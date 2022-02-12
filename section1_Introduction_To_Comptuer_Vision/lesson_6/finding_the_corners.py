
# rho = sqrt(Gx^2 + Gy^2)
# Calc the gradient for a small window of the image using sobel-x and sobel-y operators (Without applying binary thresholding)
# Use vector addition to calculate the magnitude and direction of the total gradient from these two values
# Grad x, Grad y
# Apply this calculation as you slide th window across the image calculating the gradient of each window.
# When a big variation in the direction and magnitide or the gradient has been detected, a corner has been found 

import matplotlib.pyplot as plt
import numpy as np
import cv2

# %matplotlib inline

# Read in the image
image = cv2.imread('images/waffle.jpg')

# Make a copy of the image
image_copy = np.copy(image)

# Change color to RGB (from BGR)
image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)

plt.imshow(image_copy)

###################################################################################################

# Convert to grayscale
gray = cv2.cvtColor(image_copy, cv2.COLOR_RGB2GRAY)
gray = np.float32(gray)

# Detect corners 
dst = cv2.cornerHarris(gray, 2, 3, 0.04)

# Dilate corner image to enhance corner points
dst = cv2.dilate(dst,None)

plt.imshow(dst, cmap='gray')


####################################################################################################

## TODO: Define a threshold for extracting strong corners
# This value vary depending on the image and how many corners you want to detect
# Try changing this free parameter, 0.1, to be larger or smaller ans see what happens
# higher constant i.e. >0.1 results in less matches 
# lower  constant i.e. <0.1 results in more matches
thresh = 0.05*dst.max()

# Create an image copy to draw corners on
corner_image = np.copy(image_copy)

# Iterate through all the corners and draw them on the image (if they pass the threshold)
for j in range(0, dst.shape[0]):
    for i in range(0, dst.shape[1]):
        if(dst[j,i] > thresh):
            # image, center pt, radius, color, thickness
            cv2.circle( corner_image, (i, j),1, (0,255,0), 1)

plt.imshow(corner_image)
