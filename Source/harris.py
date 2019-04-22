# Built-in libs
import numpy as np
from matplotlib import pyplot as plt
# Built-out libs
import filter as flt

# Cách 1: Non-maximum surpression va threshold
# Function to find local mixima
def Non_maximum_suppression(img):
    #Pre-Define indices of neighbors of pixel[i][j] to speed up when computing
    # xidx = np.array([-1, -1, -1, 0, 0, 1, 1, 1]) #Vertical axe: relative neighbor for x coordinate
    # yidx = np.array([-1, 0, 1, -1, 1, -1, 0, 1]) #Horizontal axe: relative neighbor for y coordinate
    xidx = np.array([0, -1, 0, 1]) #Vertical axe: relative neighbor for x coordinate
    yidx = np.array([-1, 0, 1, 0]) #Horizontal axe: relative neighbor for y coordinate
    # Get size of img
    iH, iW = img.shape

    # Init outImg
    outImg = np.zeros(img.shape, np.uint8)

    for i in range(1, iH - 1):
        for j in range(1, iW - 1):
            # If img[i][j] is not local maximum then assign 0
            if np.any(img[i][j] < img[i + xidx, j + yidx]):
                outImg.itemset((i, j), 0)
            else:
                outImg.itemset((i, j), img[i][j])
            
    return outImg

# Find feature points that response value > threshold
def thresholding(img, ratio = 0.1):
    # Compute threshold
    threshold = ratio * np.max(img)

    # Find feature by threshold image
    idx, idy = np.nonzero(img > threshold)

    # Init ouImg
    outImg = np.zeros(img.shape, np.uint8)

    # Asign feature point equal 255
    outImg[idx, idy] = 255

    return outImg            

# Cach 2: Threshold Hessian and Sort Hessian matrix descend base on response value 
# to get N strongest feature points with given minumum distance
def get_feature_points(Hessian, ratio = 0.1, minDist = 10):
    # Assume that each corner must separate at least 10 distance.
    # It mean that corner can appear within 10 distance. It avoids point will be densier in region with higher contrast
    # Compute threshold
    threshold = ratio * np.max(Hessian)

    # Find feature points above a certain threshold
    idx, idy = np.nonzero(Hessian > threshold)
    # Get value of pixel[idx, idy]
    candidateValue = Hessian[idx, idy]

    # Sort candidateValue descending by sorting then reversing
    # Then function takes first strongest corner, throws away all the nearby corners in the range of minimum distance
    #  and returns N strongest corners.
    indices = np.argsort(candidateValue)[::-1]

    # The region inside are allowed location to have corner
    # Label all alowed location with 1
    allowedPos = np.ones(Hessian.shape, np.uint8)

    # allowedPos[minDist:-minDist, minDist:-minDist] = 1

    # List store feature points
    pointsX = []
    pointsY = []

    # Find best feature points
    for i in indices:
        # Get pixel index of candidate i-th
        x, y = idx[i], idy[i]
        # If [x, y] is allowed location
        if allowedPos[x][y] == 1:
            # Store x, y index
            pointsX.append(x)
            pointsY.append(y)

            startX = (x-minDist) if (x-minDist) > 0 else 0
            endX = (x+minDist) if (x+minDist) < Hessian.shape[0] else Hessian.shape[0]

            startY = (y-minDist) if (y-minDist) > 0 else 0
            endY = (y+minDist) if (y+minDist) < Hessian.shape[1] else Hessian.shape[1]

            # All pixels around [x,y] position in range of minDist = 10 will set as non-allowed location
            allowedPos[startX:endX, startY:endY] = 0

    return [pointsX, pointsY]

def plot_feature_points(img, feturePoints):
    # Show image
    plt.figure(1)
    plt.imshow(img, cmap='gray', interpolation = 'bicubic')
    plt.title('Corners'), plt.xticks([]), plt.yticks([])

    # Plot corners
    pX, pY = feturePoints
    # Note: Axes of plot and image is reverted
    plt.plot(pY, pX, 'r*')

def detectByHarris(img, _ratio = 0.1):
    # Declare filter object
    myfilter = flt.CFilter()

    # 1. Compute sobelX and sobelY derivatives
    myfilter.gaussianGenerator(sigma = 1.0)
    Ix, Iy = myfilter.detectBySobel(img)
    
    # 2. Compute product of derivatives at every pixel
    Ix2 = Ix * Ix
    Iy2 = Iy * Iy
    Ixy = Ix * Iy

    # 3. Compute Hessian matrix: Convolve 3 image above with gaussian filter
    myfilter.gaussianGenerator(sigma = 2.5)
    Sx2 = myfilter.smoothenImage(Ix2)
    Sy2 = myfilter.smoothenImage(Iy2)
    Sxy = myfilter.smoothenImage(Ixy)

    # 4. Compute the response of detector at each pixel
    # Compute Det(H), Trace(H)
    Hdet = Sx2 * Sy2 - Sxy ** 2 
    Htr = Sx2 + Sy2
    
    # # Hessian response function: R = Det(H) - k(Trace(H))^2
    # # Harris corner detector - Harris and Stephens (1988)
    # k = 0.06
    # R = Hdet - k * (Htr ** 2)

    # # Shi & Tomashi corner detector - 1994
    # R = (Sx2 + Sy2 - np.sqrt((Sx2-Sy2)**2 + 4*(Sxy**2)))/2

    # Harmonic mean - Brown, Szeliski, and Winder (2005)
    R = np.nan_to_num(Hdet/Htr)

    # 5. Threshold on value of R. Compute nonmax suppression
    # Find local maxima above a certain threshold and report them as detected feature
    # point locations.

    # plt.figure(2)
    # plt.imshow(R, cmap='gray', interpolation = 'bicubic')
    # plt.title('Response'), plt.xticks([]), plt.yticks([])

    # thresholdImg = thresholding(R)
    # plt.figure(4)
    # plt.imshow(thresholdImg, cmap='gray', interpolation = 'bicubic')
    # plt.title('Threshold'), plt.xticks([]), plt.yticks([])

    # supImg = Non_maximum_suppression(thresholdImg)
    # plt.figure(3)
    # plt.imshow(supImg, cmap='gray', interpolation = 'bicubic')
    # plt.title('Suppression'), plt.xticks([]), plt.yticks([])

    # plt.figure(1)
    # plot_feature_points(np.nonzero(supImg == 255))
    
    # Find good feature points
    featurePoints = get_feature_points(R, ratio = _ratio)
    
    # Plot these feature points overlaid origin img
    plot_feature_points(img, featurePoints)