import numpy as np
import matplotlib.pyplot as plt
import cv2 

image = cv2.imread("images/phone.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# Define our parameters for Canny
low_threshold = 50 
high_threshold = 100
edges = cv2.Canny(gray, low_threshold, high_threshold)

# plt.imshow(edges,cmap='gray')
# plt.show()

# Hough Transform 

rho = 1                 # 1 pixel 
theta = np.pi/180       # 1 degree
threshold = 60          # minimum number of hough space intersections it takes to find a line  
min_line_length = 90    # min line length 
max_line_gap = 9        # max gap between line segments

line_image = np.copy(image)

lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), min_line_length,max_line_gap)

for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(line_image, (x1,y1), (x2,y2), (255,0,0),2)

# plt.imshow(line_image)

f, (ax1,ax2) = plt.subplots(1,2,figsize=(20,20))
ax1.set_title('Canny')
ax1.imshow(edges, cmap='gray')
ax2.set_title('Hough')
ax2.imshow(line_image, cmap='gray')

plt.show()