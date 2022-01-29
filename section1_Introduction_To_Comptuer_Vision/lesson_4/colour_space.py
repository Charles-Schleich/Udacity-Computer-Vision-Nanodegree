import cv2;
import matplotlib.pyplot as plt;
import numpy as np;

image = cv2.imread("images/water_balloons.jpg")
print("image type ", type(image))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

print("Image dimensions:", image.shape)

r = image[:,:,0]
g = image[:,:,1]
b = image[:,:,2]

if False:
    f, (ax1,ax2,ax3) = plt.subplots(3,1,figsize=(20,10))
    ax1.set_title('Red')
    ax1.imshow(r, cmap='gray')
    ax2.set_title('Green')
    ax2.imshow(g, cmap='gray')
    ax3.set_title('Blue')
    ax3.imshow(b, cmap='gray')

# Convert RGB to HSV 

hsv = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
h = hsv[:,:,0]
s = hsv[:,:,1]
v = hsv[:,:,2]

if False : 
    f, (ax1,ax2,ax3) = plt.subplots(3,1,figsize=(20,10))

    ax1.set_title('Hue')
    ax1.imshow(h, cmap='gray')
    ax2.set_title('Saturation')
    ax2.imshow(s, cmap='gray')
    ax3.set_title('Value')
    ax3.imshow(v, cmap='gray')
    plt.show()

lower_hue = np.array([160,0,0]) 
upper_hue = np.array([180,255,255])

lower_pink = np.array([180,0,100]) 
upper_pink = np.array([255,255,230])

mask_rgb = cv2.inRange(image,lower_pink,upper_pink)

masked_image = np.copy(image)
masked_image[mask_rgb==0] = [0,0,0]

# plt.imshow(masked_image)
# plt.show()

mask_hsv = cv2.inRange(hsv,lower_hue,upper_hue)
masked_image = np.copy(image)
masked_image[mask_hsv==0] = [0,0,0]

plt.imshow(masked_image)
plt.show()
