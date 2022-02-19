# The ORB algorithm is scale invariant. This means that it is able to detect objects in images regardless of their size. 
# To see this, we will now use our Brute-Force matcher to match points between the training image and a query image that is a ¼ the size of the original training image. 

import cv2
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [14.0,7.0]

image1 = cv2.imread("./images/face.jpeg")
image2 = cv2.imread("./images/faceQS.png")

training_image = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB);
query_image = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB);

# Display the images
# plt.subplot(121)
# plt.title('Training Image')
# plt.imshow(training_image)
# plt.subplot(122)
# plt.title('Query Image')
# plt.imshow(query_image)
# plt.show()

training_gray = cv2.cvtColor(training_image, cv2.COLOR_BGR2GRAY)
query_gray = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)
orb = cv2.ORB_create(1000, 2.0)

keypoints_train, descriptors_train = orb.detectAndCompute(training_gray, None)
keypoints_query, descriptors_query = orb.detectAndCompute(query_gray, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

matches = bf.match(descriptors_train, descriptors_query)
matches = sorted(matches, key = lambda x : x.distance)

result = cv2.drawMatches(training_gray, keypoints_train, query_gray, keypoints_query, matches[:30], query_gray, flags = 2)

# Print the shape of the training image
print('\nThe Training Image has shape:', training_gray.shape)
#Print the shape of the query image
print('The Query Image has shape:', query_gray.shape)
# Print the number of keypoints detected in the training image
print("\nNumber of Keypoints Detected In The Training Image: ", len(keypoints_train))
# Print the number of keypoints detected in the query image
print("Number of Keypoints Detected In The Query Image: ", len(keypoints_query))
# Print total number of matching points between the training and query images
print("\nNumber of Matching Keypoints Between The Training and Query Images: ", len(matches))


plt.title('Best Matching Points')
plt.imshow(result)
plt.show()
