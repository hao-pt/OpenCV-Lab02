import numpy as np
import scipy.integrate as sci_int
import math
import cv2
from matplotlib import pyplot as plt

def plot_feature_points(img, feturePoints):
    # Show image
    plt.figure(1)
    plt.imshow(img, cmap='gray', interpolation = 'bicubic')
    plt.title('Gray-scale image'), plt.xticks([]), plt.yticks([])
    # Plot feature points
    pX, pY = feturePoints
    # Note: Axes of plot and image is reverted
    plt.plot(pY, pX, '*')

# Read img
grayImg = cv2.imread(r"C:\Users\Tien Hao\Google Drive\Junior\TGMT\TH\DataSet\TestImages\02.jpg", cv2.IMREAD_GRAYSCALE)

# Convert type as np.float
grayf = np.float32(grayImg)
# Call cv2.cornerHarris function
dst = cv2.cornerHarris(grayf,2,3,0.04)

# Threshold image
x, y = np.nonzero(dst > 0.05*dst.max())

plot_feature_points(grayImg, (x, y))

plt.show()
