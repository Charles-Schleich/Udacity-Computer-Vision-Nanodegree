import cv2;
import matplotlib.pyplot as plt;
import numpy as np;

image = cv2.imread("images/pizza_bluescreen.jpg")
print("image type ", type(image), 
      "\nDimensions", image.shape )

image_copy = np.copy(image)
image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
                    # [r,g,b]
lower_blue = np.array([0,0,200])
upper_blue = np.array([50,75,255])

mask = cv2.inRange(image_copy,lower_blue, upper_blue)

masked_image = np.copy(image_copy)
masked_image[mask!=0] = [0,0,0]
plt.imshow(mask, cmap='gray')
plt.show()
# plt.imshow(masked_image);
# plt.show()

background_image = cv2.imread("images/space_background.jpg")
background_image = cv2.cvtColor(background_image, cv2.COLOR_BGR2RGB)
# plt.imshow(background_image);
# plt.show()

height = image_copy.shape[0]
width = image_copy.shape[1]

# 
crop_background = background_image[0:height,0:width,:]
crop_background[mask==0] = [0,0,0]

pizza_space = crop_background+masked_image
plt.imshow(pizza_space);
plt.show()
