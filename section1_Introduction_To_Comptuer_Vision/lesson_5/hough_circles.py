import numpy as np
import matplotlib.pyplot as plt
import cv2

# %matplotlib inline

# Read in the image
# image = cv2.imread('images/circles.jpg')
image = cv2.imread('images/round_farms.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Gray and blur
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

# plt.imshow(gray_blur, cmap='gray')

# for drawing circles on
circles_im = np.copy(image)

## TODO: use HoughCircles to detect circles
# right now there are too many, large circles being detected
# try changing the value of maxRadius, minRadius, and minDist
circles = cv2.HoughCircles(gray_blur, cv2.HOUGH_GRADIENT, 1, 
                           param1=70,
                           param2=11,
                           minDist=45,
                           minRadius=22,
                           maxRadius=28)

# convert circles into expected type
circles = np.uint16(np.around(circles))
# draw each one
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(circles_im,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(circles_im,(i[0],i[1]),2,(0,0,255),3)
    
plt.imshow(circles_im)
plt.show()
print('Circles shape: ', circles.shape)
    