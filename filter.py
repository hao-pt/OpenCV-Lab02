# Built-in lib
import numpy as np
import cv2
from skimage import exposure, img_as_ubyte
import math
from matplotlib import pyplot as plt

# Built-out lib
import convolution as myconv

def gaussianXFunction(size, sigma):
    # Kernel radius
    kernelRadius = size // 2
    # x will range in [-kernelRadius, kernelRadius]
    x = np.array([range(-kernelRadius, kernelRadius + 1)], np.float64).reshape(size, 1)
    # Calculate gaussian kernel X by gaussian function
    twoSquareSigma = 2 * (sigma**2)
    x_2 = x*x
    gaussX = (1/math.sqrt(math.pi*twoSquareSigma)) * np.exp(-x_2/twoSquareSigma)
    return gaussX

# Find threshold base on ratio of two threshold
# In this case, we pick sigma = 0.033 base on experience when testing often give stable result
def thresholdSeeking(img, sigma = 0.033):
    # Get median (or can get mean instead)
    med = np.median(img)
    
    #Find lowThreshold and highThreshold base on sigma
    highThreshold = math.ceil(med * (0.1 + sigma))
    lowThreshold = math.ceil(med * (0.1 - sigma))

    return lowThreshold, highThreshold

# Function to find local mixima
def Non_maximum_suppression(img):
    #Pre-Define indices of neighbors of pixel[i][j] to speed up when computing
    xidx = np.array([-1, -1, -1, 0, 0, 1, 1, 1]) #Vertical axe: relative neighbor for x coordinate
    yidx = np.array([-1, 0, 1, -1, 1, -1, 0, 1]) #Horizontal axe: relative neighbor for y coordinate

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

    # Assume that margin = 10 dont have corner.
    # The region inside (exclude margin) are allowed location to have corner 
    allowedPos = np.zeros(Hessian.shape, np.uint8)
    # Label all alowed location with 1
    allowedPos[minDist:-minDist, minDist:-minDist] = 1

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

            # All pixels around [x,y] position in range of minDist = 10 will set as non-allowed location
            allowedPos[(x-minDist):(x+minDist), (y-minDist):(y+minDist)] = 0

    return [pointsX, pointsY]

def plot_feature_points(feturePoints):
    pX, pY = feturePoints
    # Note: Axes of plot and image is reverted
    plt.plot(pY, pX, '*')

class CFilter:
    def __init__(self):
        #Gx: vertical
        #Gy: horizontal
        self.prewittKernel = {'Gx': np.array([[-1, 0, 1],
                                              [-1, 0, 1],
                                              [-1, 0, 1]], np.int),
                              'Gy': np.array([[-1, -1, -1],
                                              [0, 0, 0],
                                              [1, 1, 1]], np.int)}

        self.sobelKernel = {'Gx': np.array([[-1, 0, 1], [-2, 0, 2],
                                            [-1, 0, 1]], np.int),
                            'Gy': np.array([[-1, -2, -1],
                                            [0, 0, 0],
                                            [1, 2, 1]], np.int)}

        # GaussX, GaussY
        self.gaussKernelX = self.gaussKernelY = 0
        # Negative kernel of laplacian (3x3)
        self.laplacianKernel = np.array([[-1, -1, -1],
                                         [-1, 8, -1],
                                         [-1, -1, -1]], np.int)

        self.log5Kernel = np.array([[0, 0, 1, 0, 0],
                                    [0, 1, 2, 1, 0],
                                    [1, 2, -16, 2, 1],
                                    [0, 1, 2, 1, 0],
                                    [0, 0, 1, 0, 0]], np.int)


    def smoothenImage(self, img):
        # Declare CMyConvolution() obj
        conv = myconv.CMyConvolution()
        # Convolve 2 times to reduce time complexity
        conv.setKernel(self.gaussKernelY)
        blurImg = conv.convolution(img)
        conv.setKernel(self.gaussKernelX)
        blurImg = conv.convolution(blurImg)
        return blurImg

    def gaussianGenerator(self, size, sigma):
        # Separate gaussian into 2 direction: vertical & horizontal
        # Especially, both vertical & horizontal filter are symmetric
        # GaussX
        gaussX = gaussianXFunction(size, sigma)
        # GaussY is a transpose matrix of gaussX because they are symmetric
        gaussY = gaussX.transpose()

        # Normalize gaussian kernel by averaging
        sumX = np.sum(gaussX)
        sumY = np.sum(gaussY)
        # Round it 4 decimals
        gaussX = np.round(gaussX/sumX, 4)
        gaussY = np.round(gaussY/sumY, 4)

        self.gaussKernelX = gaussX
        self.gaussKernelY = gaussY

        return gaussX, gaussY

    def detectBySobel(self, img):
        # Declare CMyConvolution() object
        conv = myconv.CMyConvolution()
        # Smoothen image to have blur the noise and the detail of edges
        img = self.smoothenImage(img)
        # Convole with vertical kernel
        conv.setKernel(self.sobelKernel['Gx'])
        verticalImage = conv.convolution(img)
        # Convole with horizontal kernel
        conv.setKernel(self.sobelKernel['Gy'])
        horizontalImage = conv.convolution(img)
        
        return (verticalImage, horizontalImage)

    def detectByHarris(self, img):
        # Generate gassian filter
        self.gaussianGenerator(5, 1.0)
        # 1. Compute sobelX and sobelY derivatives
        Ix, Iy = self.detectBySobel(img)
        
        # 2. Compute product of derivatives at every pixel
        Ix2 = Ix * Ix
        Iy2 = Iy * Iy
        Ixy = Ix * Iy

        # 3. Compute Hessian matrix: Convolve 3 image above with gaussian filter
        self.gaussianGenerator(7, 1.5)
        Sx2 = self.smoothenImage(Ix2)
        Sy2 = self.smoothenImage(Iy2)
        Sxy = self.smoothenImage(Ixy)

        # 4. Compute the response of detector at each pixel
        # Compute Det(H), Trace(H)
        Hdet = Sx2 * Sy2 - Sxy ** 2 
        Htr = Sx2 ** 2 + Sy2 ** 2
        
        # Hessian response function: R = Det(H) - k(Trace(H))^2
        k = 0.06
        R = Hdet - k * (Htr ** 2)
        # R = np.true_divide(Hdet, Htr)

        # Threshold on value of R. Compute nonmax suppression
        # Find local maxima above a certain threshold and report them as detected feature
        # point locations.
        # supImg = Non_maximum_suppression(R)

        # plt.figure(2)
        # plt.imshow(R, cmap='gray', interpolation = 'bicubic')
        # plt.title('Response'), plt.xticks([]), plt.yticks([])

        # plt.figure(3)
        # plt.imshow(supImg, cmap='gray', interpolation = 'bicubic')
        # plt.title('Suppression'), plt.xticks([]), plt.yticks([])

        # corImg = thresholding(supImg)

        featurePoints = get_feature_points(R, ratio = 0.01)
        plot_feature_points(featurePoints)

        # return corImg
        


