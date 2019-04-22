import numpy as np
import scipy.integrate as sci_int
import math
import cv2
from matplotlib import pyplot as plt

def plot_feature_points(img, feturePoints):
    # Show image
    plt.figure(1)
    plt.imshow(img, cmap='gray', interpolation = 'bicubic')
    plt.title('Corners'), plt.xticks([]), plt.yticks([])
    # Plot feature points
    pX, pY = feturePoints
    # Note: Axes of plot and image is reverted
    plt.plot(pY, pX, 'r*')

#Measure time
e1 = cv2.getTickCount()

# Read img
grayImg = cv2.imread(r"C:\Users\Tien Hao\Google Drive\Junior\TGMT\TH\DataSet\TestImages\05.jpg", cv2.IMREAD_GRAYSCALE)

# Convert type as np.float
grayf = np.float32(grayImg)
# Call cv2.cornerHarris function
dst = cv2.cornerHarris(grayf,2,3,0.06)

# Threshold image
x, y = np.nonzero(dst > 0.03*dst.max())

# Plot feature points
plot_feature_points(grayImg, (x, y))

e2 = cv2.getTickCount()
time = (e2 - e1)/cv2.getTickFrequency()
print('Time: %.2f(s)' %(time))

plt.show()


# # Declare sift() obj
# sift = cv2.xfeatures2d.SIFT_create()
# # Sift detector
# keypoints = sift.detect(grayImg, None)

# # Flags will have draw cirle with scale of keypoint and its orientation
# img = cv2.drawKeypoints(grayImg, keypoints, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# cv2.imwrite('sift_keypoints.jpg',img)






