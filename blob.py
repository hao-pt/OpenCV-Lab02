import filter as flt
import numpy as np
import matplotlib.pyplot as plt
import math

# -----------------------Define some constants-----------------------
CONSTANT_K = math.sqrt(2) # Use for computing: nextSigma = k * sigma
# Use for keypoints localization
CONTRAST_THRESHOLD = 0.03 # Eliminate extrema with low contrast
CURVATURE_THRESHOLD = 10

class CBlob:
    def __init__(self, _no_scale_lv = 9):
        self.no_scale_lv = _no_scale_lv
        self.LoG_pyramid = [] # Store all LoG images
        self.initSigma = 1.0
        self.keypoints = [] # store keypoints
        # Ratio of 2 eigenvalue
        self.ratio_threshold = (CURVATURE_THRESHOLD + 1)**2 / CURVATURE_THRESHOLD

    def create_log_pyramid(self, img):
        # Build scale-space pyramid
        pyramid = []
        # Define lambda expression to compute next_sigma
        next_sigma = lambda k, sigma: k*sigma

        # Declare myfilter obj
        myfilter = flt.CFilter()

        for i in range(self.no_scale_lv):
            # Compute current sigma
            curSigma = next_sigma(CONSTANT_K**i, self.initSigma)
            # Generate LoG filter
            myfilter.LoG_generator(curSigma)
            # Convolve image with that filter
            logImg = myfilter.detectByLoG(img)
            # Padding logImg with constant value 0 to find extrema
            logImg = np.pad(logImg, (1, 1), mode = 'constant')
            # Squaring the response of blob to make respose strong positive to high contrast
            logImg = np.square(logImg)

            # Store in pyramid
            pyramid.append(logImg)
        
        # Store pyramid in class with numpy type
        self.LoG_pyramid = np.array(pyramid)
    
    def detectBlobByLoG(self, img):
        # Create scale-space pyramid with 9 LoG images
        self.create_log_pyramid(img)
        # Find maxima of squared Laplacian response in scale-space
        keypoints = []
        
        # Get size of original image
        iH, iW = img.shape
        
        # # Traverse through each image to find maximum
        # for x in range(1, iH - 1):
        #     for y in range(1, iW - 1):
        #         # Take a region around each pixel of each LoG image with 9*3*3 all scale to compare
        #         roiScale = self.LoG_pyramid[:, x - 1: x + 1 + 1, y-1:y+1+1]
        #         # Find maximum point
        #         max = np.amax(roiScale)

        #         # Check high contrast
        #         if max >= CONTRAST_THRESHOLD:
        #             # Find index of maximum point in roiScale
        #             # iz: iz-th layer of scale-space
        #             iz, ix, iy = np.unravel_index(roiScale.argmax(), roiScale.shape)
        #             # Push to keypoints list
        #             # Store location and scale of this keypoint
        #             keypoints.append((x + ix - 1, y + iy - 1, CONSTANT_K**iz*self.initSigma))

        # # Eliminate some duplicate keypoint because we get max of 9*3*3 neighbors at each time 
        # self.keypoints = list(set(keypoints))

        # Pre-Define indices of neighbors of pixel[i][j] to speed up when computing
        xidx = np.array([-1, -1, -1, 0, 0, 1, 1, 1, 0]) #Vertical axe: relative neighbor for x coordinate
        yidx = np.array([-1, 0, 1, -1, 1, -1, 0, 1, 0]) #Horizontal axe: relative neighbor for y coordinate

        # Traverse throush scale-space
        # Skip the topmost and lowermost. Because they dont have enough neighbor to compare
        for iz in range(1, self.no_scale_lv - 1):
            cur_img = self.LoG_pyramid[iz]
            prev_img = self.LoG_pyramid[iz - 1]
            next_img = self.LoG_pyramid[iz + 1]
            # Traverse LoG image to find maxima
            for x in range(1, iH):
                for y in range(1, iW):
                    # Check maxima
                    '''
                    One pixel in an image is compared with its 8 neighbours as well as 9 pixels 
                    in next scale and 9 pixels in previous scales. 
                    '''
                    if (np.all(cur_img[x][y] >= cur_img[x + xidx[:-1], y + yidx[:-1]]) \
                            and np.all(cur_img[x][y] >= prev_img[x + xidx, y + yidx]) \
                            and np.all(cur_img[x][y] >= next_img[x + xidx, y + yidx])):
                        
                        '''
                        * Getting rid of low-contrast and edge keypoints
                        * Rejection:
                        - Low contrast: Threshold intensities (|D(x)| < 0.03 in term of range[0, 1]) then it is rejected. 
                        - Lie on edge: Use concept similar with harris 
                        but In SIFT, efficiency is increased by just calculating the ratio of these two eigenvalues.
                            Tr(H)**2 / Det(H) < (r+1)**2 / r , where r = 10
                        Which eliminates keypoints that have a ratio between the principal curvatures greater than 10.
                        * Therefore, the remains is strong keypoints.
                        '''
                        
                        # Check if it is low contrast
                        if cur_img[x][y] < CONTRAST_THRESHOLD:
                            continue
                        
                        # Check edge
                        # 2x2 Hessian matrix (H) computed at the location and scale of the keypoint:
                        # Because i use x as vertical and y as horizontal, 
                        # when computing derivative of x and y need to revert x, y synstax
                        dxx = cur_img[x][y - 1] + cur_img[x][y + 1] - 2*cur_img[x][y]
                        dyy = cur_img[x - 1][y] + cur_img[x + 1][y] - 2*cur_img[x][y]
                        dxy = (cur_img[x - 1][y - 1] + cur_img[x + 1][y + 1] - cur_img[x - 1][y + 1] 
                        - cur_img[x + 1][y - 1]) / 4.0

                        # Compute trace and det of H
                        trH = dxx + dyy
                        detH = dxx*dyy - dxy**2

                        # Compute curvature_ratio
                        curvature_ratio = np.nan_to_num(trH**2 / detH)

                        if curvature_ratio <= self.ratio_threshold:
                            # Store location and scale of this keypoint
                            keypoints.append((x - 1, y - 1, CONSTANT_K**iz*self.initSigma))
                        

        # Store keypoints to class
        self.keypoints = keypoints

    def plotBlob(self, img):
        fig, axes = plt.subplots()
        
        # Set title and show img
        axes.set_title("Blob detector used Laplacian of Gaussian")
        axes.imshow(img, interpolation='nearest', cmap="gray")

        for blob in self.keypoints:
            # Get location and sigma of blob
            x, y, sigma = blob
            # Note: radious of blob = sigma * sqrt(2)
            radious = sigma * CONSTANT_K
            circle = plt.Circle((y, x), radious, color='red', linewidth=1.5, fill=False)
            axes.add_patch(circle) # Add circle to axis

        # Turn off axis
        axes.set_axis_off()
    



