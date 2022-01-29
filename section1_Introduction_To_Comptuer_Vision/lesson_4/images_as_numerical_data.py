import numpy as np
import matplotlib.image as mpimg # reading images

import matplotlib.pyplot as plt
import cv2

image = mpimg.imread('images/waymo_car.jpg')
print("image Dimensions:", image.shape)

gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

plt.imshow(gray_image,cmap='gray')
# plt.show()

x = 180
y = 375
# gray_image[y,x]= 255
print(gray_image[y,x])
# plt.imshow(gray_image, cmap='gray')
# plt.show()

max_val = np.amax(gray_image)
min_val = np.amin(gray_image)

print('Max: ', max_val)
print('Min: ', min_val)

tiny_smile = [
    [0,0,255,0,255,0,0],
    [0,0,255,0,255,0,0],
    [255,0,0,0,0,0,255],
    [0,255,0,0,0,255,0],
    [0,0,255,255,255,0,0],
    [0,0,0,0,0,0,0],
    ]
plt.matshow(tiny_smile, cmap='gray')
plt.show()