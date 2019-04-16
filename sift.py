# Built-in libs
import numpy as np
import math
# Built-out libs
import filter as flt

# Define some constants
CONTRAST_THRESHOLD = 0.03
CURVATURE_THRESHOLD = 10

class CSift:
    def __init__(self, img = []):
        self.srcImg = img
        # Paper author choose no. octave = 4, no. scale = 5, k = sqrt(2)
        self.no_octave = 4 # number of octave
        self.no_scale = 5 # number of scale level per octave
        # Constant value k
        self.k = np.sqrt(2)

        # Sigma table
        self.sigma_table = []
        
        # Scale-space pyramid with multiple gausssian blur images
        self.layer_pyramid = []
        # DoG images
        self.dog_pyramid = []
        # Key points
        self.keypoints = []
        # Ratio of 2 eigenvalue
        self.ratio_threshold = (CURVATURE_THRESHOLD + 1)**2 / CURVATURE_THRESHOLD

    
    # Create prior smoothing image before building scale-space for an octave
    # Paper author choose sigma = 1.6 which provides close to optimal repeatability
    def prior_smoothing(self, sigma = 1.6):
        myfilter = flt.CFilter()
        myfilter.gaussianGenerator(sigma = sigma)
        self.srcImg = myfilter.smoothenImage()
        return self.srcImg

    # Create scale-space table (sigma table)
    def create_scale_space_table(self):
        # Initial sigma
        sigma = 1/2
        # Define lambda expression to compute next_sigma in this current octave
        next_sigma = lambda k, sigma: k*sigma
        
        # Sigma table
        self.sigma_table = []

        # Compute first octave and scale levels
        first_octave = [next_sigma(self.k , sigma)]
        for j in range(1, self.no_scale):
            sigma = first_octave[j - 1]
            first_octave.append(next_sigma(self.k, sigma))
        self.sigma_table.append(first_octave)

        # Compute rest octave and scale level of each octave
        for i in range(1, self.no_octave):
            ith_octave = [2*x for x in self.sigma_table[i - 1]]
            self.sigma_table.append(ith_octave)

        self.sigma_table = np.array(self.sigma_table)
        return self.sigma_table      

    # Build scale-space pyramid generate all gaussian-blur image for each octave and DoG image
    def build_scale_space_pyramid(self):
        # Declare filter obj
        myfilter = flt.CFilter()

        # Build scale-space pyramid
        pyramid = []
        for i, octave in enumerate(self.sigma_table):
            ith_octave = []
            for j, sigma in enumerate(octave):
                # Generate gaussian kernel with given sigma
                myfilter.gaussianGenerator(sigma = sigma)
                # Append gauss-blur image into list
                ith_octave.append(myfilter.smoothenImage(self.srcImg))
            # Append ith octave list into pyramid list
            pyramid.append(ith_octave)
        
        # Store layer_pyramid
        self.layer_pyramid = pyramid

        DoG = []
        # Build DoG pyramid by subtracting 2 consecutive gauss-blur image
        for i, octave in enumerate(self.layer_pyramid):
            ith_octave = []
            for j in range(1, self.no_scale):
                # Append DoG image of ith octave into list 
                ith_octave.append(octave[j] - octave[j - 1])
            DoG.append(ith_octave)
        # Store dog_pyramid
        self.dog_pyramid = DoG

    # Find potential keypoints (Locate maxima/minima in DoG images)
    def scale_space_extrema_detection(self):
        # 1. Create sigma table
        self.create_scale_space_table()
        
        # 2. Build scale-space
        self.build_scale_space_pyramid()
        
        # 3. Locate maxima/minima in DoG images
        '''
        One pixel in an image is compared with its 8 neighbours as well as 9 pixels in next scale 
        and 9 pixels in previous scales. 
        If it is a local extrema (smallest or greatest), it is a potential keypoint.
        '''
        # Pre-Define indices of neighbors of pixel[i][j] to speed up when computing
        xidx = np.array([-1, -1, -1, 0, 0, 1, 1, 1, 0]) #Vertical axe: relative neighbor for x coordinate
        yidx = np.array([-1, 0, 1, -1, 1, -1, 0, 1, 0]) #Horizontal axe: relative neighbor for y coordinate

        self.keypoints = []

        for i, dog in enumerate(self.dog_pyramid):
            # Skip the topmost and lowermost. Because they dont have enough neighbor to compare
            for j in range(1, len(dog) - 1):
                # Get dog image size
                iH, iW = dog[j].shape
                
                # Dog images: current scale, prev scale, next scale
                cur_img = dog[j]
                prev_img = dog[j - 1]
                next_img = dog[j + 1]

                # Traverse image
                for x in range(1, iH - 1):
                    for y in range(1, iW - 1):
                        # Check maxima 
                        if np.all(cur_img[x][y] >= cur_img[x + xidx[:-1], y + yidx[:-1]]) \
                                and np.all(cur_img[x][y] >= prev_img[x + xidx, y + yidx]) \
                                and np.all(cur_img[x][y] >= next_img[x + xidx, y + yidx]):
                            self.keypoints.append([x, y, i, j]) # Store location: (x, y), ith octave and jth dog image
                        
                        # check minima
                        elif np.all(cur_img[x][y] <= cur_img[x + xidx[:-1], y + yidx[:-1]]) \
                                and np.all(cur_img[x][y] <= prev_img[x + xidx, y + yidx]) \
                                and np.all(cur_img[x][y] <= next_img[x + xidx, y + yidx]):
                            self.keypoints.append([x, y, i, j]) # Store location: (x, y), ith octave and jth dog image
        
        print(len(self.keypoints))
    
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
    def keypoints_localization(self):
        n = len(self.keypoints)
        i = 0
        while i < n:
            # Get position of feature point
            x, y = self.keypoints[i][:2]
            # Get octave and dog location
            ioct = self.keypoints[i][2]
            idog = self.keypoints[i][3]

            # Get dog image
            curImg = self.dog_pyramid[ioct][idog]

            # Check low constrast
            if math.fabs(curImg[x][y]) < CONTRAST_THRESHOLD:
                self.keypoints.pop(i) # remove this keypoint
                n = n - 1
                continue

            # Check edge
            # 2x2 Hessian matrix (H) computed at the location and scale of the keypoint:
            dxx = curImg[x - 1][y] + curImg[x + 1][y] - 2*curImg[x][y]
            dyy = curImg[x][y - 1] + curImg[x][y + 1] - 2*curImg[x][y]
            dxy = (curImg[x - 1][y - 1] + curImg[x + 1][y + 1] - curImg[x - 1][y + 1] - curImg[x + 1][y - 1]) / 4

            # Compute trace and det of H
            trH = dxx + dyy
            detH = dxx*dyy - dxy**2

            # Compute curvature_ratio
            curvature_ratio = np.nan_to_num(trH**2 / detH)

            if curvature_ratio > self.ratio_threshold:
                self.keypoints.pop(i)
                n = n - 1
                continue
            
            i = i + 1

        print(len(self.keypoints))

    def detectBySift(self):
        # 1. Find potential keypoints (Locate maxima/minima in DoG images)
        self.scale_space_extrema_detection()
        # 2. Getting rid of low-contrast and edge keypoints
        self.keypoints_localization()
    
