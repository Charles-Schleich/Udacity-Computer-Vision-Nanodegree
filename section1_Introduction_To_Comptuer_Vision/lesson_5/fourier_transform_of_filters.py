# fouter_transform_of_filters.py

import numpy as np
import matplotlib.pyplot as plt
import cv2

# %matplotlib inline

# Define gaussian, sobel, and laplacian (edge) filters

gaussian = (1/9)*np.array([[1, 1, 1],
                           [1, 1, 1],
                           [1, 1, 1]])

sobel_x= np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])

sobel_y= np.array([[-1,-2,-1],
                   [0, 0, 0],
                   [1, 2, 1]])

# laplacian, edge filter
laplacian=np.array([[0, 1, 0],
                    [1,-4, 1],
                    [0, 1, 0]])

filters = [gaussian, sobel_x, sobel_y, laplacian]
filter_name = ['gaussian','sobel_x', \
                'sobel_y', 'laplacian']


# perform a fast fourier transform on each filter
# and create a scaled, frequency transform image
f_filters = [np.fft.fft2(x) for x in filters]
fshift = [np.fft.fftshift(y) for y in f_filters]
# fshift = f_filters

frequency_tx = [np.log(np.abs(z)+1) for z in fshift]

# display 4 filters
for i in range(len(filters)):
    plt.subplot(2,2,i+1),plt.imshow(frequency_tx[i],cmap = 'gray')
    plt.title(filter_name[i]), plt.xticks([]), plt.yticks([])

plt.show()

## TODO: load in an image, and filter it using a kernel of your choice
## apply a fourier transform to the original *and* filtered images and compare them

image = cv2.imread('images/city_hall.jpg')
print(image.dtype)
image_copy = np.copy(image)
image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

gaussian = (1/9)*np.array([[1, 1, 1],
                           [1, 1, 1],
                           [1, 1, 1]])
sobel_x= np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])
sobel_y= np.array([[-1,-2,-1],
                   [0, 0, 0],
                   [1, 2, 1]])
laplacian=np.array([[0, 1, 0],
                    [1,-4, 1],
                    [0, 1, 0]])

image_gaussian = cv2.filter2D(gray, -1, gaussian)
image_sobel_x = cv2.filter2D(gray, -1, sobel_x)
image_sobel_y = cv2.filter2D(gray, -1, sobel_y)
image_laplacian = cv2.filter2D(gray, -1, laplacian)

fig=plt.figure(figsize=(8,8), dpi= 300, facecolor='w', edgecolor='k')
plt.subplot(4,2,1),plt.imshow(gray,cmap = 'gray')
plt.subplot(4,2,2),plt.imshow(image_gaussian,cmap = 'gray')

plt.subplot(4,2,3),plt.imshow(gray,cmap = 'gray')
plt.subplot(4,2,4),plt.imshow(image_laplacian,cmap = 'gray')

plt.subplot(4,2,5),plt.imshow(gray,cmap = 'gray')
plt.subplot(4,2,6),plt.imshow(image_sobel_y,cmap = 'gray')

plt.subplot(4,2,7),plt.imshow(gray,cmap = 'gray')
plt.subplot(4,2,8),plt.imshow(image_sobel_x,cmap = 'gray')

plt.show()

# plt.title(filter_name[i]), plt.xticks([]), plt.yticks([])
   
