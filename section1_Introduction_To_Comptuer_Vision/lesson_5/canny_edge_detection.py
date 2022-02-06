# canny edge detection 

from cv2 import cvtColor
import numpy as np
import matplotlib.pyplot as plt
import cv2

image = cv2.imread('images/sunflower.jpg')
image_copy = np.copy(image)

image_copy = cv2.cvtColor(image_copy,cv2.COLOR_BGR2RGB)

gray = cv2.cvtColor(image_copy, cv2.COLOR_RGB2GRAY);
# plt.imshow( gray,cmap='gray')
# plt.show()

wide = cv2.Canny(gray, 30,100)
tight = cv2.Canny(gray,200,240)

f, (ax1,ax2) = plt.subplots(1,2,figsize=(20,10))

ax1.set_title('wide')
ax1.imshow(wide, cmap='gray')

ax2.set_title('tight')
ax2.imshow(tight, cmap='gray')
plt.show()
