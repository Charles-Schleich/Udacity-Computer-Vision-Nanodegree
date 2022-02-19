import cv2
import matplotlib.pyplot as plt

# Set the default figure size
plt.rcParams['figure.figsize'] = [14.0, 7.0]
# Load the training image
image1 = cv2.imread('./images/face.jpeg')
image2 = cv2.imread('./images/faceRN5.png')

#
query_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
training_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

# Display the images
# plt.subplot(121)
# plt.imshow(training_gray, cmap = 'gray')
# plt.title('Gray Scale Training Image')
# plt.subplot(122)
# plt.imshow(query_gray, cmap = 'gray')
# plt.title('Query Image')
# plt.show()

# Set the parameters of the ORB algorithm by specifying the maximum number of keypoints to locate and
# the pyramid decimation ratio
orb = cv2.ORB_create(1000, 1.3)

keypoints_train, descriptors_train = orb.detectAndCompute(training_gray, None)
keypoints_query, descriptors_query = orb.detectAndCompute(query_gray, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
matches = bf.match(descriptors_train, descriptors_query)
matches = sorted(matches, key = lambda x : x.distance)
result = cv2.drawMatches(training_gray, keypoints_train, query_gray, keypoints_query, matches[:100], query_gray, flags = 2)

print("Number of Keypoints Detected In The Training Image: ", len(keypoints_train))
print("Number of Keypoints Detected In The Query Image: ", len(keypoints_query))
print("\nNumber of Matching Keypoints Between The Training and Query Images: ", len(matches))

plt.title('Best Matching Points')

plt.subplot(131)
plt.imshow(training_gray, cmap = 'gray')
plt.title('Gray Scale Training Image')
plt.subplot(132)
plt.imshow(query_gray, cmap = 'gray')
plt.title('Query Image')
plt.subplot(133)
plt.imshow(result, cmap = 'gray')
plt.imshow(result)
plt.show()
