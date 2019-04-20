import numpy as np
import scipy.integrate as sci_int
import math
import cv2
from matplotlib import pyplot as plt

# def plot_feature_points(feturePoints):
#     pX, pY = feturePoints
#     # Note: Axes of plot and image is reverted
#     plt.plot(pY, pX, '*')

# # Read img
# img = cv2.imread("checkerboard.png")
# # Gray-scale image
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # Convert type as np.float
# grayf = np.float32(gray)
# # Call cv2.cornerHarris function
# dst = cv2.cornerHarris(grayf,2,3,0.04)

# # Threshold image
# x, y = np.nonzero(dst > 0.1*dst.max())

# plt.figure(1)
# plt.imshow(gray, cmap='gray', interpolation = 'bicubic')
# plt.title('Corner detection'), plt.xticks([]), plt.yticks([])

# plot_feature_points((x, y))

# plt.show()

a = np.array([[1, 2, 3], [5, 3, 3]])
b = np.array([[5, 3, -1], [-3, 10, 3]])
c = np.array([a, b])
print(c[1])

print(max([1, 2, 3, -1, 5, 10]))