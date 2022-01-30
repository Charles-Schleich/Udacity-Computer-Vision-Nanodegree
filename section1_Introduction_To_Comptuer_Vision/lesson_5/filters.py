import numpy as np
import matplotlib.pyplot as plt
import cv2


image = cv2.imread("./images/city_hall.jpg")

image_copy = np.copy(image)

image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
# plt.imshow(image_copy)
# plt.show()

gray = cv2.cvtColor(image_copy, cv2.COLOR_RGB2GRAY)

# IMPORTANT: Kernels must add up to zero otherwise they 
# will be lightenning or darkenning an image
# literally adding or subtracting power from an image 
# Sobel filter
sobel_x = np.array([[-1,0,1],
                   [-2,0,2],
                   [-1,0,1],
                  ])

sobel_y = np.array([[-1,-2,-1],
                   [0,0,0],
                   [1,2,1],
                  ])

# Perfom convolution using Filter2D, 
#                            (img, bit-depth, kernel)
filtered_image = cv2.filter2D(gray, -1, sobel_y);
plt.imshow(filtered_image, cmap='gray')
# plt.show()

# Create binary image
retval, binary_image = cv2.threshold(filtered_image, 100,255, cv2.THRESH_BINARY)
plt.imshow(binary_image, cmap='gray')
plt.show()

