import cv2 
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [20,10]

image = cv2.imread('./images/face.jpeg')

training_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

training_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

plt.subplot(121)
plt.title('Original Training Image')
plt.imshow(training_image)
plt.subplot(122)
plt.title('Gray Scale Training Image')
plt.imshow(training_gray, cmap='gray')
# plt.show()

# Locating keypoints in image using ORB
# Below is the ORB function and its defaults
# cv2.ORB_create(nfeatures = 500,
#                scaleFactor = 1.2,
#                nlevels = 8,
#                edgeThreshold = 31,
#                firstLevel = 0,
#                WTA_K = 2,
#                scoreType = HARRIS_SCORE,
#                patchSize = 31,
#                fastThreshold = 20)
# Parameters:
#     nfeatures     - int
#     Determines the maximum number of features (keypoints) to locate.
#     scaleFactor   - float
#     Pyramid decimation ratio, must be greater than 1. ORB uses an image pyramid to find features, therefore you must provide the scale factor between each layer in the pyramid and the number of levels the pyramid has. A scaleFactor = 2 means the classical pyramid, where each next level has 4x less pixels than the previous. A big scale factor will diminish the number of features found.
#     nlevels       - int
#     The number of pyramid levels. The smallest level will have a linear size equal to input_image_linear_size/pow(scaleFactor, nlevels).
#     edgeThreshold - int
#     The size of the border where features are not detected. Since the keypoints have a specific pixel size, the edges of images must be excluded from the search. The size of the edgeThreshold should be equal to or greater than the patchSize parameter.
#     firstLevel    - int
#     This parameter allows you to determine which level should be treated as the first level in the pyramid. It should be 0 in the current implementation. Usually, the pyramid level with a scale of unity is considered the first level.
#     WTA_K         - int
#     The number of random pixels used to produce each element of the oriented BRIEF descriptor. The possible values are 2, 3, and 4, with 2 being the default value. For example, a value of 3 means three random pixels are chosen at a time to compare their brightness. The index of the brightest pixel is returned. Since there are 3 pixels, the returned index will be either 0, 1, or 2.
#     scoreType     - int
#     This parameter can be set to either HARRIS_SCORE or FAST_SCORE. The default HARRIS_SCORE means that the Harris corner algorithm is used to rank features. The score is used to only retain the best features. The FAST_SCORE produces slightly less stable keypoints, but it is a little faster to compute.
#     patchSize     - int
#     Size of the patch used by the oriented BRIEF descriptor. Of course, on smaller pyramid layers the perceived image area covered by a feature will be larger.

import copy
plt.rcParams['figure.figsize'] = [14.0,7.0]
orb = cv2.ORB_create(200,2.0)

keypoints, descriptor = orb.detectAndCompute(training_gray,None)

# Create copies of the training image to draw our keypoints on
keyp_without_size = copy.copy(training_image)
keyp_with_size = copy.copy(training_image)

# Draw the keypoints without size or orientation on one copy of the training image 
cv2.drawKeypoints(training_image, keypoints, keyp_without_size, color=(0,255,0))

# Draw the keypoints with size and orientation on the other copy of the training image
cv2.drawKeypoints(training_image, keypoints, keyp_with_size, flags= cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# cv2.drawKeypoints(training_image, keypoints, keyp_with_size, flags= cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

plt.subplot(121)
plt.title('Keypoints Without Size or Orientation')
plt.imshow(keyp_without_size)

plt.subplot(122)
plt.title("Keypoints With Size and Orientation")
plt.imshow(keyp_with_size)
# plt.show()

print("Number of Keypoints detected", len(keypoints))

# Next we use the Brute Force for matching

# cv2.BFMatcher(normType = cv2.NORM_L2,
# 		 	  crossCheck = false)
# Parameters:
#       normType
#       Specifies the metric used to determine the quality of the match. By default, normType = cv2.NORM_L2, which measures the distance between two descriptors. However, for binary descriptors like the ones created by ORB, the Hamming metric is more suitable. The Hamming metric determines the distance by counting the number of dissimilar bits between the binary descriptors. When the ORB descriptor is created using WTA_K = 2, two random pixels are chosen and compared in brightness. The index of the brightest pixel is returned as either 0 or 1. Such output only occupies 1 bit, and therefore the cv2.NORM_HAMMING metric should be used. If, on the other hand, the ORB descriptor is created using WTA_K = 3, three random pixels are chosen and compared in brightness. The index of the brightest pixel is returned as either 0, 1, or 2. Such output will occupy 2 bits, and therefore a special variant of the Hamming distance, known as the cv2.NORM_HAMMING2 (the 2 stands for 2 bits), should be used instead. Then, for any metric chosen, when comparing the keypoints in the training and query images, the pair with the smaller metric (distance between them) is considered the best match.
#   crossCheck - bool 
#   A Boolean variable and can be set to either True or False. Cross-checking is very useful for eliminating false matches. Cross-checking works by performing the matching procedure two times. The first time the keypoints in the training image are compared to those in the query image; the second time, however, the keypoints in the query image are compared to those in the training image (i.e. the comparison is done backwards). When cross-checking is enabled a match is considered valid only if keypoint A* in the training image is the best match of keypoint *B in the query image and vice-versa (that is, if keypoint B* in the query image is the best match of point *A in the training image).

plt.rcParams['figure.figsize'] = [14.0, 7.0]

# Load training image
image1 = cv2.imread('./images/face.jpeg')

# Load query image
image2 = cv2.imread('./images/face.jpeg')

training_image = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

query_image = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

plt.subplot(121)
plt.title('Training Image')
plt.imshow(training_image)
plt.subplot(122)
plt.title('Query Image')
plt.imshow(query_image)
plt.show()

training_gray = cv2.cvtColor(training_image,cv2.COLOR_BGR2GRAY)
query_gray = cv2.cvtColor(query_image,cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create(1000,2.0)

