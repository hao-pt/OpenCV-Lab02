# Built-in lib
import math
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import exposure, img_as_ubyte
import scipy.integrate as sci_int

# Built-out lib
import convolution as myconv


def integrand(x, sigma):
    # Calculate gaussian kernel X by gaussian function
    twoSquareSigma = 2 * (sigma**2)
    twoPi = 2 * math.pi
    x_2 = x*x
    gaussX = (1/(math.sqrt(twoPi) * sigma)) * np.exp(-x_2/twoSquareSigma)
    return gaussX

# Compute gaussianX integral
def gaussX_int(f, sigma, a, b):
    return sci_int.quad(f, a, b, args = sigma)[0]

# Calculate vertical gaussian X
def gaussianXFunction(size, sigma):
    minX = -size/2
    maxX = size/2
    step = 1

    # a, b
    a_range = np.arange(minX, maxX - step + 1.0, step)
    b_range = np.arange(minX + step, maxX + 1.0, step)

    # Vectorize gaussX_int function
    vec_gaussX_int = np.vectorize(gaussX_int)

    # Replicate gaussX_int function for each a, b
    gaussX = vec_gaussX_int(integrand, sigma, a_range, b_range)
    
    return gaussX.reshape(-1, 1)

# Find threshold base on ratio of two threshold
# In this case, we pick sigma = 0.033 base on experience when testing often give stable result
def thresholdSeeking(img, sigma = 0.033):
    # Get median (or can get mean instead)
    med = np.median(img)
    
    #Find lowThreshold and highThreshold base on sigma
    highThreshold = math.ceil(med * (0.1 + sigma))
    lowThreshold = math.ceil(med * (0.1 - sigma))

    return lowThreshold, highThreshold




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
        
        gaussX = gaussX/sumX
        gaussY = gaussY/sumY

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


