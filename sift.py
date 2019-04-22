# Built-in libs
import numpy as np
import math
import matplotlib.pyplot as plt
# Built-out libs
import filter as flt
import blob

# -----------------------Define some constants-----------------------
# Use for keypoints localization
CONTRAST_THRESHOLD = 0.03
CURVATURE_THRESHOLD = 10 
# Use for assigning orientation
NUM_BINS = 36

class CSift:
    def __init__(self, img = [], _no_octave = 4, _no_scale = 5):
        self.srcImg = img
        # Paper author choose no. octave = 4, no. scale = 5, k = sqrt(2)
        self.no_octave = _no_octave # number of octave
        self.no_scale = _no_scale # number of scale level per octave
        self.no_extrema_images = self.no_scale // 2 # Number of extrema image without topmost and botmost in dog images
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
        sigma = 1.0
        # Define lambda expression to compute next_sigma in this current octave
        next_sigma = lambda k, sigma: k*sigma
        
        # Sigma table
        self.sigma_table = []

        # Compute first octave and scale levels
        first_octave = [sigma]
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
                dogImg = octave[j] - octave[j - 1]
                # Squaring the response of blob to make respose strong positive to high contrast
                dogImg = np.abs(dogImg)
                # Append DoG image of ith octave into list 
                ith_octave.append(dogImg)
            DoG.append(ith_octave)

        # Store dog_pyramid
        self.dog_pyramid = DoG

    # Find candidate keypoints
    def find_candidate_keypoints(self):
        # 1. Create sigma table
        self.create_scale_space_table()
        
        # 2. Build scale-space
        self.build_scale_space_pyramid()
        
        # 3. Find candidate keypoints:
        # Locate maxima/minima in DoG images
        # Get rid of low contrast and edge
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
            # So there are just 2 extrema images
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
                            # Check low constrast
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
                                # Store location: (x, y), ith octave and jth dog image
                                self.keypoints.append([x, y, i, j])

        print(len(self.keypoints))
    
    def compute_magnitude_and_orientation(self):
        # Init magnitude and orientation array (Store all images over scale space of keypoints)
        # The scale of the keypoint is used to select the Gaussian smoothed image, L, with the closest scale
        # that mean we choose the closest scale of keypoint in gaussian-blur images
        # So there are 4 octaves and 2 closest scales to store
        magnitude = []
        orientation = [] # Note orientation now range in [-PI, PI]

        # Traverse through out all octaves and 2 closest scale level of keypoints
        for i in range(self.no_octave):
            # Init magnitude_per_oct and orient_per_octave
            magnitude_per_oct = []
            orient_per_oct = []
            # Note: There are 2 extrema images that have keypoints such as DoG images No.1, No.2 (index start from 0)
            for j in range(self.no_extrema_images):
                # Get gaussian-blur image L at current scale space i, j + 1
                L = self.layer_pyramid[i][j+1]

                # Get size of image
                iH, iW = L.shape

                # Init magnitude image and orient image of L
                magL = np.zeros(L.shape) 
                oriL = np.zeros(L.shape)

                # Traverse this image (Dont account for pixels outside border = 1)
                for x in range(1, iH - 1):
                    for y in range(1, iW - 1):
                        # Compute magnitude of this pixel
                        dx = L[x][y + 1] + L[x][y - 1]
                        dy = L[x + 1][y] + L[x - 1][y]
                        magL[x][y] = math.sqrt(dx**2 + dy**2)

                        # Compute orientation of this pixel
                        oriL[x][y] = math.atan2(dy, dx)

                # Store magL and oriL per octave
                magnitude_per_oct.append(magL)       
                orient_per_oct.append(oriL)
            
            # Store gradient magnitude and orientation per octave
            magnitude.append(magnitude_per_oct)
            orientation.append(orient_per_oct)
        
        return (magnitude, orientation)

    # Assign an orientation to each keypoints. This is invariant to rotation
    def assign_orientation(self):
        # 1. Compute magnitude and orientation for each gaussian-blur image L
        magnitude, orientation = self.compute_magnitude_and_orientation()

        print(np.array(magnitude).shape, np.array(orientation).shape)

        ''' 2. Orient assignment by building histogram of 36 bin and small region around each keypoints
            to find dominent orientation of each keypoints.

            - An orientation histogram with 36 bins covering 360 degrees is created. 
            (10 degree per bin and range in [0, 360])

            - Each sample added to the histogram is weighted by its gradient magnitude and by a Gaussian-weighted 
            circular window with a σ = 1.5*sigma, where sigma is the scale of the keypoint. 
            - The window size (small region), or the "orientation collection region", is equal to the size of 
            the kernel for Gaussian Blur of amount 1.5*sigma.

            - Using histogram, The highest peak in the histogram is detected.
                + If there is only one peak at a certain location, it is assigned to the keypoint. 
                + If there are multiple peaks above the 80% mark at a certain location, they are all converted into 
                a new keypoint (with their respective orientations). It's orientation is equal to the other peak.
                Therefore, it creates keypoints with same location and scale, but different directions.

            - Finally, a parabola is fit to the 3 histogram values closest to each peak to interpolate the peak 
            position for better accuracy. 
            Particularly, It uses quadratic term y = ax^2 + bx + c then take first order derivative to find a 
            maximum bin of 3 closest bin
        '''
        
        degreePerBin = 360 / NUM_BINS # Each bin is 10 degree

        keypoints = [] # List to store keypoint info after assigning orientation

        # Traverse all keypoints list
        for i in range(len(self.keypoints)):
            # Get position of feature point
            x, y = self.keypoints[i][:2]
            # Get octave and dog location
            ioct = self.keypoints[i][2]
            idog = self.keypoints[i][3]

            # Get scale level (sigma) because it chooses closest scale between 2 gaussian-blur image
            sigma = self.sigma_table[ioct][idog]

            # Get index of gradient magnitude and orientation
            ii, jj = ioct, idog - 1
            # Get gradient magnitude and orientation image at ii, jj
            magImg = magnitude[ii][jj]
            oriImg = orientation[ii][jj]

            # Weighted image between gradien magnitude and gaussian with σ = 1.5*sigma
            myfilter = flt.CFilter()
            myfilter.gaussianGenerator(sigma = 1.5*sigma) # Generate gaussian
            weightedImg = myfilter.smoothenImage(magImg)

            # Get kernel size of gaussian-blur
            ksize = 2 * math.ceil(2 * (1.5*sigma)) + 1

            # Get size of image
            iH, iW = oriImg.shape

            # Init histogram orientation with 36 bins
            hist_ori = np.zeros(NUM_BINS)

            #Go through all region around keypoint called "Small region around each keypoint"
            #To build histogram with 36 bins
            for ix in range(-ksize + x, ksize + x + 1):
                for iy in range(-ksize + y, ksize + y + 1):
                    # Make sure (ix, iy) dont out of range
                    if ix < 0 or ix >= iH or iy < 0 or iy >= iW:
                        continue
                    
                    # Turn orientation in range [0, 2pi]
                    sampleOri = oriImg[ix, iy] + math.pi
                    # Convert to degree
                    sampleOri = sampleOri * 180/math.pi
                    # Count it by weighted image and push to correspondent bin
                    hist_ori[int(sampleOri // degreePerBin)] += weightedImg[ix][iy]
            
            # startX = -ksize + x if (-ksize + x) >= 0 else 0
            # endX = ksize + x + 1 if (ksize + x + 1) <= iH else iH
            # startY = -ksize + y if (-ksize + y) >= 0 else 0
            # endY = ksize + y + 1 if (ksize + y + 1) <= iW else iW
            
            # # Take the region of interest for oriImg
            # roiOriImg = oriImg[startX:endX, startY:endY]
            # # Take the region of interest for weightedImg
            # roiWeightedImg = weightedImg[startX:endX, startY:endY]

            # # Turn orientation in range [0, 2pi]
            # sampleOri = roiOriImg + math.pi
            # # Convert to degree
            # sampleOri = sampleOri * 180/math.pi
            
            # # Flatten sampleOri into 1-d array
            # bin_list = np.ravel(sampleOri // degreePerBin).astype(int)
            # # Flatten weighted image into 1-d array
            # flatWeighted = roiWeightedImg.ravel()
            # # Count it by weighted image and push to correspondent bin
            # for i, b in enumerate(bin_list):
            #     hist_ori[b] = flatWeighted[i]
            
            # print(hist_ori)

            # Find max_peak
            max_peak = np.amax(hist_ori)

            # Find all magnitude and orientation of this keypoint
            ori = []
            mag = []

            # Find all peak above 80% of max_peak
            z_indices = np.nonzero(hist_ori > (max_peak*0.8))
            
            # Traverse through peaks to find dominent orientation
            for z in z_indices:
                # fit parabola to the 3 histogram values closest to each peak
                # To find maximum peak by taking this equation: y = ax^2 + bx + c
                # Now, (x2, y2) is the peak then (x1, y1), (x3, y3) are left and right bin
                
                # check value for X and Y
                # If peak is 1st bin, (x1, y1) will equal the right most bin (NUM_BINS - 1, hist_ori[NUM_BINS - 1]) 
                # and vice versa
                z = int(z)
                if z == 0:
                    x_values = np.array([NUM_BINS - 1, z, z + 1])
                    y_values = np.array([hist_ori[NUM_BINS - 1], hist_ori[z], hist_ori[z + 1]])
                elif z == NUM_BINS - 1:
                    x_values = np.array([z - 1, z, 0])
                    y_values = np.array([hist_ori[z - 1], hist_ori[z], hist_ori[0]])
                else:
                    x_values = np.array([z - 1, z, z + 1])
                    y_values = np.array([hist_ori[z - 1], hist_ori[z], hist_ori[z + 1]])
                
                # We have y = ax^2 + bx + c
                # And we have 3 points (x1, y1), (x2, y2), (x3, y3)
                # y1 = ax1^2 + bx1 + c
                # y2 = ax2^2 + bx2 + c
                # y3 = ax3^2 + bx3 + c
                # Vectorize this equation, we have Y = Xw
                # Y = [y1, y2, y3]'
                # X = [[x1^2, x1, 1]', [x2^2, x2, 1]', [x3^2, x3, 1]']
                # So, w = inv(X)Y where w = [a, b, c]'
                X = np.array([
                    [x_values[0]**2, x_values[1]**2, x_values[2]**2], 
                    [x_values[0], x_values[1], x_values[2]],
                    [1, 1, 1]])
                Y = y_values.T
                w = (np.linalg.pinv(X)).dot(Y)

                # Now, take 1st derivative to find maximum peak: 0 = 2ax + b -> x = -b/2a
                x0 = -w[1] / (2*w[0])

                while x0 > NUM_BINS:
                    x0 -= NUM_BINS
                while x0 < 0:
                    x0 += NUM_BINS

                # Convert to degree
                x0 = x0 * (2*math.pi / NUM_BINS)

                # Turn x0 back to range[-pi, pi]
                x0 -= math.pi

                # Store this dominent orientation
                ori.append(x0)
                mag.append(hist_ori[z])
        
            # Save this keypoint with multiple orientations and magnitudes at the same location (x, y)
            keypoints.append([x, y, ori, mag, ii, jj])

        print(len(keypoints))
        print(len(self.keypoints))
        # Store in class
        self.keypoints = keypoints

    # Plot blob used matplotlib
    def plotBlob(self):
        fig, axes = plt.subplots()
        
        # Set title and show img
        axes.set_title("Blob detector used Difference of Gaussian")
        axes.imshow(self.srcImg, interpolation='nearest', cmap="gray")

        for blob in self.keypoints:
            # Get location and sigma of blob
            x, y, zx, zy = blob
            # Get sigma
            sigma = self.sigma_table[zx][zy]
            # Note: radious of blob = sigma * sqrt(2)
            radious = sigma * self.k
            circle = plt.Circle((y, x), radious, color='red', linewidth=1.5, fill=False)
            axes.add_patch(circle) # Add circle to axis

        # Turn off axis
        axes.set_axis_off()

    # Blob detector used DoG
    def detectBlobByDoG(self):
        '''
        There are 2 steps:
            1. Find potential keypoints (Locate maxima/minima in DoG images)
                In this case, i just locate maxima because i take absolute of the blob's respoonse 
            2. Getting rid of low-contrast and edge keypoints'''
        self.find_candidate_keypoints()
        
        
    def read_features(self, filename = ""):
        '''
        Read features properties from file
        Each row contain first 4 values - location (coordinates, scale, dominent angle) 
        and followed by 128 values of corresponding descriptor 
        '''
        # Read feature properties with numpy loadtxt
        # Each row present for 1 feature
        features = np.loadtxt(filename)
        # Return locations and descriptors
        return features[:,:4], features[:,4:]

    def write_features(self, filename, locations, descriptors):
        """Save features location and descriptor into file"""
        np.savetxt(filename, np.hstack(locations, descriptors))

    def plot_features(self, img, locations, isDrawRichKeypoints = True):
        '''
        isDrawRichKeypoints: default True
        Use for drawing keypoints with size and orientation and vice versa.
        '''
        # Draw keypoint with size and orientation
        if isDrawRichKeypoints:
            kpImg = cv2.drawKeypoints(img, locations, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        else:
            kpImg = cv2.drawKeypoints(img, locations, color = (0,0,255)) # Keypoints are drawn with red color
        
        # Show image and keypoints
        plt.figure()
        plt.imshow(kpImg, cmap='gray', interpolation = 'bicubic')
        plt.title('Sift keypoints'), plt.xticks([]), plt.yticks([])
        
        # Save image
        # cv2.imwrite('sift_keypoints.jpg',img)

    def draw_matches(self, train_img, train_locations, query_img, query_locations, matches):
        '''
        Locations include: coordinates (x, y), scale and angle
        matches: are number of matched keypoints between train_img and query_img
        '''
        # flags = 2: Single keypoints will not be drawn.
        matchedImg = cv2.drawMatches(train_img, train_locations, query_img, query_locations, matches, flags = 2)

        # Show image with matching points
        plt.title('Matching Points between 2 images')
        plt.imshow(matchedImg)
        plt.show()

        # Print the number of matching keypoints between 2 images
        print("The number of matching keypoints between 2 images are %d." %len(matches))

    # Sift detector
    def detectBySift(self):

        '''
        1. Find candidate keypoints
        There are 2 steps:
            1. Find potential keypoints (Locate maxima/minima in DoG images)
                In this case, i just locate maxima because i take absolute of the blob's respoonse 
            2. Getting rid of low-contrast and edge keypoints'''
        self.find_candidate_keypoints()
        # 2. Assign an orientation to each keypoints. This is invariant to rotation
        self.assign_orientation()

