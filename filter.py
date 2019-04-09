# Built-in lib
import numpy as np
import cv2
from skimage import exposure, img_as_ubyte
import math
import matplotlib.pyplot as plt
# Built-out lib
import convolution as myconv
import stack

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

# Laplacian of gaussian (LoG) function
def laplacianOfGaussian(size, sigma):
    # Kernel radius
    kernelRadius = size // 2
    # x will range in [-kernelRadius, kernelRadius]
    x = np.array([range(-kernelRadius, kernelRadius + 1)],
                 np.float64).reshape(size, 1)
    X = np.tile(x, (1, size))  # Copy and put (size) vectors onto x
    # y will range in [-kernelRadius, kernelRadius]
    y = np.array([range(-kernelRadius, kernelRadius + 1)], np.float64)
    Y = np.tile(y, (size, 1))  # Copy and stack (size) row vectors on y
    # Calculate LoG kernel
    sigma_4 = sigma**4
    twoSquareSigma = 2 * (sigma**2)
    S = (np.power(X, 2) + np.power(Y, 2))/twoSquareSigma
    LoG = (-1/(math.pi*sigma_4))*(1 - S)*np.exp(-S)
    return (LoG * (2**size)).astype(np.int)

# Thinning multiple-pixel of edge into single pixel
def non_max_surpression(img, theta):
    Angle = theta * 180/np.pi # Convert it to degree
    Angle[Angle < 0] += 180 # To reduce the number of angle comparsion. 
    #So we are just dealing with 4 main angles: 0, 45, 90, 135
    
    #Get size of image
    iH, iW = img.shape

    #Init surpressImage with 0-element
    surpressImg = np.zeros((iH, iW), np.uint8)

    for y in range(1, iH - 1):
        for x in range(1, iW - 1):
            
            t1 = 255
            t2 = 255

            #Angle 0
            if (0 <= Angle[y][x] < 22.5) or (157.5 <= Angle[y][x] <= 180):
                t1 = img[y][x - 1]
                t2 = img[y][x + 1]
            #Angle 45
            elif (22.5 <= Angle[y][x] < 67.5):
                t1 = img[y - 1][x + 1]
                t2 = img[y + 1][x - 1]
            #Angel 90
            elif (67.5 <= Angle[y][x] < 112.5):
                t1 = img[y - 1][x]
                t2 = img[y + 1][x]
            #Angle 135
            elif (112.5 <= Angle[y][x] < 157.5):
                t1 = img[y - 1][x - 1]
                t2 = img[y + 1][x + 1]

            #Check if pixel[y][x] >= t1 and pixel[y][x] >= t2, asign to surpressImg. Otherwise is 0
            if (img[y][x] >= t1) and (img[y][x] >= t2):
                surpressImg.itemset((y, x), img[y][x])
            else:
                surpressImg.itemset((y, x), 0)
        
    
    return surpressImg

# Find threshold base on ratio of two threshold
# In this case, we pick sigma = 0.033 base on experience when testing often give stable result
def thresholdSeeking(img, sigma = 0.033):
    # Get median (or can get mean instead)
    med = np.median(img)
    
    #Find lowThreshold and highThreshold base on sigma
    highThreshold = math.ceil(med * (0.1 + sigma))
    lowThreshold = math.ceil(med * (0.1 - sigma))

    return lowThreshold, highThreshold

# Utility that is to find strong and weak pixel related to edge
def thresholding(img, lowThreshold, highThreshold):
    # Any pixel have intensity greater than high threshold which are 'sure edge'
    sureEdgeX, sureEdgeY = np.nonzero(img >= highThreshold)
    # Any pixel below low threshold which are non-edge. Dont matter
	# Those are lie between these two threshold are classified: edge or non-edge according to 
    # their connectivity with strong pixel called weak pixel
    weakEdgeX, weakEdgeY = np.nonzero((lowThreshold <= img) & (img < highThreshold))
    
    #Define strong and weak value to measure by myself
    minValue = 25
    maxValue = 255

    #Init thresholdingImg with 0-element
    thresholdingImg = np.zeros(img.shape, np.uint8)
    
    #Now image just consists two pixel intensity value: strong and weak (240, 25)
    thresholdingImg[sureEdgeX, sureEdgeY] = maxValue
    thresholdingImg[weakEdgeX, weakEdgeY] = minValue

    return thresholdingImg, minValue, maxValue

# Final step of Canny is to transform weak pixel to strong pixel of edge if it has at least 1 neighbor 
# which is sure edge
def hysteresis(img, minValue, maxValue):
    #Pre-Define indices of neighbors of pixel[i][j] to speed up when computing
    xidx = np.array([-1, -1, -1, 0, 0, 1, 1, 1]) #Vertical axe: relative neighbor for x coordinate
    yidx = np.array([-1, 0, 1, -1, 1, -1, 0, 1]) #Horizontal axe: relative neighbor for y coordinate
    
    #Get size of img
    iH, iW = img.shape

    #Copy img to edgeImg
    edgeImg = np.copy(img)

    # # Use scanline: Faster but less detail than floodfill
    # #Now traverse img to transform pixel from weak to strong pixel
    # for i in range(1, iH - 1):
    #     for j in range(1, iW - 1):
    #         # If pixel[i][j] has at least 1 neighbor which is 'sure edge'
    #         if (edgeImg[i][j] == minValue):
    #             if (np.any(edgeImg[xidx + i, yidx + j] == maxValue)):
    #                 edgeImg.itemset((i, j), maxValue)
    #             else:
    #                 edgeImg.itemset((i, j), 0)

    # Use floodfill: fast, more detail than scanline
    
    # Init label matrix to check if pixel[i][j] visited
    label = np.zeros(img.shape, np.uint8)
    for x in range(1, iH):
        for y in range(1, iW):
            if edgeImg[x][y] != 0 and label[x][y] == 0:
                # Instantiate stack
                S = stack.CStack()
                # Push (x, y) index into stack
                S.push((x, y))
                #Flood fill until stack is empty
                while not(S.empty()):
                    i, j = S.pop()
                    if edgeImg[i][j] == minValue:
                        # If pixel[i][j] has at least 1 neighbor which is 'sure edge'
                        if (np.any(edgeImg[xidx + i, yidx + j] == maxValue)):
                            edgeImg.itemset((i, j), maxValue)
                        else:
                            edgeImg.itemset((i, j), 0)
                    # Mark it equal 1 when pass it
                    label[i][j] = 1
                    # Find 8-neighbor which are weak pixel
                    for z in range(8):
                        nx = xidx[z] + i
                        ny = yidx[z] + j
                        # Check if pixle[nx][ny]:  is not border and dont traverse before
                        if (1 <= nx <= iH - 1) and (1 <= ny <= iW - 1) and label[nx][ny] == 0:
                            # Check if pixle[nx][ny] is not 'non-edge'
                            if edgeImg[nx][ny] != 0:
                                S.push((nx, ny))

    return edgeImg
                            

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

        self.weightedAvgKernel = np.array([[1, 2, 1],
                                           [2, 4, 2],
                                           [1, 2, 1]], np.float64) * 1.0/16
        self.meanKernel = np.ones((3, 3), np.int) * 1.0/9
        self.gaussKernelX = self.gaussKernelY = 0
        # Negative kernel of laplacian (3x3)
        self.laplacianKernel = np.array([[-1, -1, -1],
                                         [-1, 8, -1],
                                         [-1, -1, -1]], np.int)
        self.logKernel = 0
        self.log5Kernel = np.array([[0, 0, 1, 0, 0],
                                    [0, 1, 2, 1, 0],
                                    [1, 2, -16, 2, 1],
                                    [0, 1, 2, 1, 0],
                                    [0, 0, 1, 0, 0]], np.int)

        # self.log9Kernel = np.array([[0, 1, 1, 2, 2, 2, 1, 1, 0],
        #                             [1, 2, 4, 5, 5, 5, 4, 2, 1],
        #                             [1, 4, 5, 3, 0, 3, 5, 4, 1],
        #                             [2, 5, 3, -12, -24, -12, 3, 5, 2],
        #                             [2, 5, 0, -24, -40, -24, 0, 5, 2],
        #                             [2, 5, 3, -12, -24, -12, 3, 5, 2],
        #                             [1, 4, 5, 3, 0, 3, 5, 4, 1],
        #                             [1, 2, 4, 5, 5, 5, 4, 2, 1],
        #                             [0, 1, 1, 2, 2, 2, 1, 1, 0]], np.int)

    def smoothenImage(self, img, kernelName):
        conv = myconv.CMyConvolution()
        if kernelName == 'mean':
            conv.setKernel(self.meanKernel)
        elif kernelName == 'weighted avg':
            conv.setKernel(self.weightedAvgKernel)
        elif kernelName == 'gauss':
            # Convolve 2 times to reduce time complexity
            conv.setKernel(self.gaussKernelY)
            blurImg = conv.convolution(img)
            conv.setKernel(self.gaussKernelX)
            blurImg = conv.convolution(blurImg)
            return blurImg

        blurImg = conv.convolution(img)
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

    # Laplacian generator
    def logGenerator(self, size, sigma):
        self.logKernel = laplacianOfGaussian(size, sigma)
        return self.logKernel

    def detectBySobel(self, img, norm = False):
        # Declare CMyConvolution() object
        conv = myconv.CMyConvolution()
        # Smoothen image to have blur the noise and the detail of edges
        img = self.smoothenImage(img, 'gauss')
        # Convole with vertical kernel
        conv.setKernel(self.sobelKernel['Gx'])
        verticalImage = conv.convolution(img, norm = norm)
        # Convole with horizontal kernel
        conv.setKernel(self.sobelKernel['Gy'])
        horizontalImage = conv.convolution(img, norm = norm)
        # Combine 2 vertical & horizontal image together to get magnitude of gradient at each point
        # |G| = sqrt(Gx^2 + Gy^2)
        # Typically, |G| = |Gx| + |Gy|
        # avoid out of range [0, 255]

        magnitudeImg = np.zeros(img.shape, dtype=np.float64)
        magnitudeImg = np.sqrt(np.power(verticalImage.astype(
            np.float64), 2) + np.power(horizontalImage.astype(np.float64), 2))

        # magnitudeImg = np.abs(verticalImage.astype(
        # np.float64)) + np.abs(horizontalImage.astype(np.float64))

        # Normalize the output image to be in range [0, 255] accurately_
        # _when it's presented in float dtype [0, 1] called 'shrinking image'
        magnitudeImg = exposure.rescale_intensity(
            magnitudeImg, out_range=(0, 1))
        # Convert dtype of image back to uint8
        magnitudeImg = img_as_ubyte(magnitudeImg)

        return (verticalImage, horizontalImage, magnitudeImg)

    def detectByPrewitt(self, img):
        # Declare CMyConvolution() object
        conv = myconv.CMyConvolution()
        # Smoothen image to have blur the noise and the detail of edges
        img = self.smoothenImage(img, 'gauss')

        # Convole with vertical kernel
        conv.setKernel(self.prewittKernel['Gx'])
        verticalImage = conv.convolution(img)
        # Convole with horizontal kernel
        conv.setKernel(self.prewittKernel['Gy'])
        horizontalImage = conv.convolution(img)
        # Combine 2 vertical & horizontal image together to get magnitude of gradient at each point
        # |G| = sqrt(Gx^2 + Gy^2)
        # Combine 2 vertical & horizontal image together to get magnitude of gradient at each point
        # avoid out of range [0, 255]
        magnitudeImg = np.zeros(img.shape, dtype=np.float64)
        magnitudeImg = np.sqrt(np.power(verticalImage.astype(
            np.float64), 2) + np.power(horizontalImage.astype(np.float64), 2))

        # Normalize the output image to be in range [0, 255] accurately_
        # _when it's presented in float dtype [0, 1] called 'shrinking image'
        magnitudeImg = exposure.rescale_intensity(
            magnitudeImg, out_range=(0, 1))
        # Convert dtype of image back to uint8
        magnitudeImg = img_as_ubyte(magnitudeImg)
        return (verticalImage, horizontalImage, magnitudeImg)

    def detectByLaplacian(self, img):
        '''
        3 steps:
            step 1:
                Gaussian smoothing
            step 2:
                Convolve smoothed image with Laplacian kernel
            step 3:
                find zero crossings in the resulting image of previous step 
        '''
        # Declare CMyConvolution() object
        conv = myconv.CMyConvolution()
        
        height = img.shape[0] # chieu cao cua anh
        width = img.shape[1] # chieu rong cua anh
        # Step 1: Gaussian smoothing
        smoothedImg = self.smoothenImage(img, 'gauss')
        destinationImg = np.zeros(img.shape, np.uint8)
        
        # Step 2: Convolve smoothed image with Laplacian kernel (finish step 1 and 2)
        conv.setKernel(self.laplacianKernel) 
        step1_step2 = conv.convolution(smoothedImg, True)
        
        #step1_step2 = conv.doConvolution(sourceImg)
    
        neighbor_pos = ((-1,0), (1,0), (0,-1), (0,1), (-1,-1), (1,1), (-1,1), (1,-1))
    
        tmpImg = step1_step2
        
        # Step 3: find zero-crossings
        
        ''' - At each pixel in the image, considering 8 neighbors of that pixel.
            - These 8 neighbors make into 4 symmetric pairs: (left, right), (up, down) and 2 diagonals.
            - Then, counting how many pairs which have different sign in their 2 elements. 
            (and the absolute value of the difference also needs to be greater than given threshold)
            - If the result is more than 1, that pixel is zero crossing.''' 
        
        for y in range(height):
            for x in range(width):
                count = 0 
                for k in range(0,8,2):
                    n1_pos = (neighbor_pos[k][0]+y, neighbor_pos[k][1]+x)
                    n2_pos = (neighbor_pos[k+1][0]+y, neighbor_pos[k+1][1]+x)
                    if 0 <= n1_pos[0] < height and 0 <= n1_pos[1] < width and 0 <= n2_pos[0] < height and 0 <= n2_pos[1] < width:
                        t = int(step1_step2[n1_pos[0], n1_pos[1]]) * step1_step2[n2_pos[0],n2_pos[1]] #Get sign
                        if t < 0:
                            if abs(int(step1_step2[n1_pos[0], n1_pos[1]]) - step1_step2[n2_pos[0],n2_pos[1]]) > 10:
                                count += 1
                        elif t == 0:
                            if step1_step2[n1_pos[0],n1_pos[1]] < 0 or step1_step2[n2_pos[0],n2_pos[1]] < 0:
                                if abs(int(step1_step2[n1_pos[0], n1_pos[1]]) - step1_step2[n2_pos[0],n2_pos[1]]) > 10:
                                    count += 1
           
                if count < 2:
                    tmpImg[y][x] = 0
                    
        tmpImg[tmpImg < 0] = abs(tmpImg[tmpImg <0]) + 30
       
        destinationImg = tmpImg.astype(np.uint8)
            
        return destinationImg

    def detectByCanny(self, img):
        # Canny edge detector:
        # 1. Blur image with deravative of gaussian
        # Declare CMyConvolution() object
        conv = myconv.CMyConvolution()
        blurImg = self.smoothenImage(img, 'gauss')


        # 2. Find magnitude and orientation of gradient
        # 	- deltaX, deltaY and magnitude of X,Y orientations (by Sobel)
        deltaX, deltaY, deltaXY = self.detectBySobel(blurImg, norm = True)
        # 	- theta matrix of edge angles:
        #       theta = arctan(deltaY/deltaX)
        theta = np.arctan2(deltaY, deltaX)
                

        # 3. Non-maximum suppression:
        # 	- Thin multi-pixel wide 'ridges' down to single pixel width
        surpressImg = non_max_surpression(deltaXY, theta)


        # 4. Linking and thresholding (hysteresis):
        # 	- Define two thresholds: low and high
    	#   - Use the high threshold to start edge curves and low threshold to continue them (edge map)

        #Find 2 thresholds
        lowThreshold, highThreshold = thresholdSeeking(img)
        #Threshold img
        thresholdingImg,minValue,maxValue = thresholding(surpressImg, lowThreshold, highThreshold)
        
        #Final step, turn weak pixel into strong pixel if it has connected with 'sure edge'
        edgeImg = hysteresis(thresholdingImg, minValue, maxValue)

        return blurImg, deltaXY, surpressImg, thresholdingImg, edgeImg
