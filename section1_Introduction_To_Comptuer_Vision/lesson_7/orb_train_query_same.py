import cv2 
import matplotlib.pyplot as plt


plt.rcParams['figure.figsize'] = [14.0, 7.0]

# Load training image
image1 = cv2.imread('./images/face.jpeg')

# Load query image
image2 = cv2.imread('./images/face.jpeg')

training_image = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

query_image = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

# plt.subplot(121)
# plt.title('Training Image')
# plt.imshow(training_image)
# plt.subplot(122)
# plt.title('Query Image')
# plt.imshow(query_image)
# plt.show()

training_gray = cv2.cvtColor(training_image,cv2.COLOR_BGR2GRAY)
query_gray = cv2.cvtColor(query_image,cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create(1000,2.0)

keypoints_train, descriptors_train = orb.detectAndCompute(training_gray,None)
keypoints_query, descriptors_query = orb.detectAndCompute(query_gray,None)

bf =  cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

# Perform the matching between the ORB descriptors of the training image and the query image
matches = bf.match(descriptors_train, descriptors_query)

# The matches with shorter distance are the ones we want. So, we sort the matches according to distance
matches = sorted(matches, key = lambda x : x.distance)

# Connect the keypoints in the training image with their best matching keypoints in the query image.
# The best matches correspond to the first elements in the sorted matches list, since they are the ones with the shorter distance. We draw the first 300 mathces and use flags = 2 to plot the matching keypoints without size or orientation.

result = cv2.drawMatches(training_gray, keypoints_train, query_gray, keypoints_query, matches[:300], query_gray, flags = 2)

# Print the number of keypoints detected in the training image
print("Number of Keypoints Detected In The Training Image: ", len(keypoints_train))
# Print the number of keypoints detected in the query image
print("Number of Keypoints Detected In The Query Image: ", len(keypoints_query))
# Print total number of matching points between the training and query images
print("Number of Matching Keypoints Between The Training and Query Images: ", len(matches))

plt.title('Best Matching Points')
plt.imshow(result)
plt.show()

