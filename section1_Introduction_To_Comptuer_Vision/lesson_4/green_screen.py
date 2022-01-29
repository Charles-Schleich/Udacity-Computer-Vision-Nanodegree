import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import numpy as np
import cv2

image = mpimg.imread("images/car_green_screen.jpg")
print("Image dimensions:", image.shape)

y_dimm = image.shape[0]
x_dimm = image.shape[1]
print("x:",x_dimm," y:",y_dimm)
# plt.imshow(image)
# plt.show()
              #        [r,g,b]
lower_green = np.array([30,80,30])
upper_green = np.array([140,255,120])

mask = cv2.inRange(image, lower_green, upper_green)
# plt.imshow(mask, cmap='gray')
# plt.show()

mask_image = np.copy(image)
mask_image [mask!=0] = [0,0,0]

# plt.imshow(mask_image,)
# plt.show()

background_image = mpimg.imread("images/sky.jpg")

# height = background_image.shape[0]
# width = background_image.shape[1]
crop_background = background_image[0:y_dimm,0:x_dimm,:]

background_image_mask= np.copy(crop_background)
background_image_mask[mask==0] = [0,0,0]

complete_image = background_image_mask+mask_image

plt.imshow(complete_image)
plt.show()