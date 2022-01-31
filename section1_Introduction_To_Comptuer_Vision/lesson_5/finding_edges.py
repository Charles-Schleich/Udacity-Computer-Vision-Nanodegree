import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import cv2
import numpy as np

# Read in the image
image = mpimg.imread('images/curved_lane.jpg')
#plt.imshow(image)

# Convert to grayscale for filtering
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
plt.imshow(gray, cmap='gray')
# plt.show()

# Create a custom kernel
# 3x3 array for edge detection
sobel_y = np.array([[ -1, -2, -1], 
                   [ 0, 0, 0], 
                   [ 1, 2, 1]])

## TODO: Create and apply a Sobel x operator
sobel_x = np.array([[ -1, 0, 1], 
                   [ -2, 0, 2], 
                   [ -1, 0, 1]])

# 
# Filter the image using filter2D, which has inputs: (grayscale image, bit-depth, kernel)  
filtered_image = cv2.filter2D(gray, -1, sobel_y)
#plt.imshow(filtered_image, cmap='gray')

blur = np.array([[ 1/9, 1/9, 1/9], 
                 [ 1/9, 1/9, 1/9], 
                 [ 1/9, 1/9, 1/9]])


# blur_5x5 = np.array([
#                  [ 1/25, 1/25, 1/25, 1/25, 1/25], 
#                  [ 1/25, 1/25, 1/25, 1/25, 1/25], 
#                  [ 1/25, 1/25, 1/25, 1/25, 1/25],
#                  [ 1/25, 1/25, 1/25, 1/25, 1/25],
#                  [ 1/25, 1/25, 1/25, 1/25, 1/25]])

blur_5x5 = np.array([
                 [ 1, 1, 1, 1, 1], 
                 [ 1, 1, 1, 1, 1], 
                 [ 1, 1, 1, 1, 1],
                 [ 1, 1, 1, 1, 1],
                 [ 1, 1, 1, 1, 1]])


(w,h) = blur_5x5.shape
elems = w*h
scale = lambda x, y : x *y
vf = np.vectorize(scale)
new_blur = vf(blur_5x5,1/elems)
print(blur_5x5)
print(new_blur)


blur_9x9 = np.ones((9,9))
(w,h) = blur_9x9.shape
elems = w*h
blur_9x9 = vf(blur_9x9,1/elems)

# 5 by 5 kernel 
# sobel_y_5x5 = np.array([
#                    [-1,-1,-2,-1,-1], 
#                    [ 0, 0, 0, 0, 0], 
#                    [ 0, 0, 0, 0, 0], 
#                    [ 0, 0, 0, 0, 0], 
#                    [ 1, 1, 2, 1, 1]])

sobel_y_7x7 = np.array([
                   [-1,-1,-2,-3,-2,-1,-1], 
                   [ 0, 0, 0, 0, 0, 0, 0], 
                   [ 0, 0, 0, 0, 0, 0, 0], 
                   [ 0, 0, 0, 0, 0, 0, 0], 
                   [ 0, 0, 0, 0, 0, 0, 0], 
                   [ 0, 0, 0, 0, 0, 0, 0], 
                   [ 1, 1, 2, 3, 2, 1, 1]])

## TODO: Create and apply a Sobel x operator
# sobel_x = np.array([[ -1, 0, 1], 
#                    [ -2, 0, 2], 
#                    [ -1, 0, 1]])

# Filter the image using filter2D, which has inputs: (grayscale image, bit-depth, kernel)  
filtered_image = cv2.filter2D(gray, -1, blur_9x9)

plt.imshow(filtered_image, cmap='gray')
plt.show()






