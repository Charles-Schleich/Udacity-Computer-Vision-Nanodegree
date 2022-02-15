import cv2
# from cv2 import TERM_CRITERIA_EPS 
import matplotlib.pyplot as plt
import numpy as np

image = cv2.imread('./images/monarch.jpg')

image = cv2.imread("images/monarch.jpg")

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


pixel_vals  = image.reshape((-1,3))
pixel_vals = np.float32(pixel_vals)


criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

k = 4
#  (pixel values, k, any labels we want, our stop criteria, number of attempts, starting centers  )
retval, labels, centers =  cv2.kmeans(pixel_vals, k , None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Convert data back to 8-bit data values 
centers  = np.uint8(centers)
segmented_data = centers[labels.flatten()]

segmented_image = segmented_data.reshape((image.shape))
labels_reshape = labels.reshape(image.shape[0], image.shape[1])
plt.imshow(segmented_image)
# plt.show()


# mask an image segment by cluster
cluster = 0 # the first cluster
masked_image = np.copy(image)
# turn the mask blue!
masked_image[labels_reshape == cluster] = [0, 0, 255]

plt.imshow(masked_image)
plt.show()