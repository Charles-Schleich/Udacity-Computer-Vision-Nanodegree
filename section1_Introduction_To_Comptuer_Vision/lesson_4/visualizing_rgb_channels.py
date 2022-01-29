# visualizing_rgb_channels.py

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image = mpimg.imread('images/wa_state_highway.jpg')

# plt.imshow(image)
# plt.show()
print(image.shape)

# Isolate RGB channels
# image 
r = image[:,:,0]
g = image[:,:,1]
b = image[:,:,2]
# original = image[:,:,:]
# e = (r + b)

# Visualize the individual color channels
f, (ax1, ax2, ax3) = plt.subplots( 3,1, figsize=(50,50))

# ax1.imshow(original, cmap='')
ax1.set_title('R channel')
ax1.imshow(r, cmap='gray')
ax2.set_title('G channel')
ax2.imshow(g, cmap='gray')
ax3.set_title('B channel')
ax3.imshow(b , cmap='gray')
plt.show()