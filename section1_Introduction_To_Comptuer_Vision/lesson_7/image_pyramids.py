import numpy as np
import matplotlib.pyplot as plt
import cv2


# Read in the image
image = cv2.imread('images/rainbow_flag.jpg')

# Change color to RGB (from BGR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# plt.imshow(image)
# plt.show()

level_1 = cv2.pyrDown(image)
level_2 = cv2.pyrDown(level_1)
level_3 = cv2.pyrDown(level_2)

f, (ax1, ax2, ax3, ax4) = plt.subplots(4,1,figsize=(20,10))

ax1.set_title('original')
ax1.imshow(image)

ax2.set_title('level_1')
ax2.imshow(level_1)
ax2.set_xlim([0, image.shape[1]])
ax2.set_ylim([image.shape[0], 0])

ax3.set_title('level_2')
ax3.imshow(level_2)
ax3.set_xlim([0, image.shape[1]])
ax3.set_ylim([image.shape[0], 0])

ax4.set_title('level_3')
ax4.imshow(level_3)
ax4.set_xlim([0, image.shape[1]])
ax4.set_ylim([image.shape[0], 0])

plt.show()
