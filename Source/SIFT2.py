# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 16:34:17 2019

@author: VÕQUỐCHUY
"""
import numpy as np
import cv2
from numpy import linalg as LA
import math
from sklearn.preprocessing import normalize
from matplotlib import pyplot as plt

class SIFT:
    def __init__(self,_img,_kernelSize,_sigma,_k,_numsOfScales,_numsOfOctaves):
        self.img = _img
        self.kernelSize = _kernelSize
        self.sigma = _sigma
        self.k = _k
        self.numsOfScales = _numsOfScales
        self.numsOfOctaves = _numsOfOctaves
        self.listOfOctaves = []
        self.m = 0
        self.theta = 0
        
    def createImgs_AtOctave(self,img):
        height,width = img.shape
        listOfImgs = []
        for i in range(self.numsOfScales):
            blurImg = cv2.GaussianBlur(img,(self.kernelSize,self.kernelSize),self.sigma*pow(self.k,i))
            listOfImgs.append((blurImg,self.sigma*pow(self.k,i)))
        return listOfImgs
    
    def createImgs_AtMultipleOctaves(self):
        # not complete: choose the start sigma for the next octave
        #listOfOctaves = [] ## Make it to be a instance variable of this class
        sigma = self.sigma
        for i in range(self.numsOfOctaves):
            #resizedImg = cv2.resize(self.img, (0,0), fx=pow(0.5,i), fy=pow(0.5,i))
            octave = self.createImgs_AtOctave(self.img)
            self.listOfOctaves.append(octave)
            sigma *= pow(self.k,(self.numsOfScales//2))
        return self.listOfOctaves
    
    def calDOG_AtMultipleOctaves(self):
        #listOfOctaves = self.createImgs_AtMultipleOctaves()
        listOfDOGImgs = []
        for octave in self.listOfOctaves:
            tmpList = []
            for i in range(1,len(octave)):
                tmpImg1 = octave[i][0].astype(np.int8)
                tmpImg2 = octave[i-1][0].astype(np.int8)
                DOGImgs = tmpImg1 - tmpImg2
                # DOGImgs[DOGImgs<0] = abs(DOGImgs[DOGImgs<0]) +50
                # DOGImgs = DOGImgs.astype(np.uint8)
                tmpList.append((DOGImgs,octave[i-1][1])) # ?? i-1 or i
            listOfDOGImgs.append(tmpList)
        return listOfDOGImgs
    
    def aux_findExtremaOnOneDOGImg(self,img_below,img_middle,img_above,C_DOG,C_edge):# return a list of tuples
        height,width = img_middle.shape
        img_below = cv2.copyMakeBorder(img_below, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
        img_middle = cv2.copyMakeBorder(img_middle, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
        img_above = cv2.copyMakeBorder(img_above, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
        
        eight_neighbors = np.array([[0,-1,-1,0,1,1,1,0,-1],
							          [0,0,1,1,1,0,-1,-1,-1]], np.int)
        returnList = []
        for i in range(1,height+1):
            for j in range(1,width+1):
                # get 27 neighbor pixels
                listOfNeighborVals = []
                for k in range(9):
                    dy,dx = eight_neighbors[0,k],eight_neighbors[1,k]
                    ny = i + dy
                    nx = j + dx
                    if 1 <= ny < height+1 and 1 <= nx < width+1:
                        listOfNeighborVals.append(img_middle[ny][nx])
                        listOfNeighborVals.append(img_below[ny][nx])
                        listOfNeighborVals.append(img_above[ny][nx])
                
                curVal = listOfNeighborVals[0]
                flag1 = True # Is Maxima?
                flag2 = True # Is Minima?
                
                for val in listOfNeighborVals:
                    if curVal < val:
                        flag1 = False
                    if curVal > val:
                        flag2 = False
                    if flag1 == False and flag2 == False:
                        break
                
                isDiscarded = False
                if flag1 == True or flag2 == True:
                    # Discarding low contrasted extrema
                    if img_middle[i,j] < 0.8*C_DOG:
                        isDiscarded = True
                    #Discarding candidate keypoints on edge
                    int_img_middle = img_middle.astype(np.int8)
                    if isDiscarded == False:
                        h11 = int_img_middle[i+1,j] + int_img_middle[i-1,j] - 2*int_img_middle[i,j]
                        h22 = int_img_middle[i,j+1] + int_img_middle[i,j-1] - 2*int_img_middle[i,j]
                        h12 = (int_img_middle[i+1,j+1] - int_img_middle[i-1,j+1] - int_img_middle[i+1,j-1] + int_img_middle[i-1,j-1])/4
                        H = np.array([[h11,h12],
                                      [h12,h22]],np.float64)
                        w, v = LA.eig(H)
                        edgeness = pow(w[0]+w[1],2)/(w[0]*w[1])
                        if edgeness > pow(C_edge+1,2)/C_edge:
                            isDiscarded = True
                            
                    if isDiscarded == False:
                        returnList.append((i,j))
                
                '''if len(listOfNeighborVals[curVal >= listOfNeighborVals]) == len(listOfNeighborVals):
                    returnList.append((i,j))
                if len(listOfNeighborVals[curVal <= listOfNeighborVals]) == len(listOfNeighborVals):
                    returnList.append((i,j))'''
        return returnList
             
    
    def findApproxKeypoints(self,C_DOG=0.015,C_edge=10): # C_DOG is threshold for discarding lwo contrasted keypoints
        # C_edge for discarding candidate keypoints on edges
        listOfDOGImgs = self.calDOG_AtMultipleOctaves()
        listOfKeypoints = []
        for octave in listOfDOGImgs: 
            tmpList = []
            for s in range(1,len(octave)-1):
                listOfExtrema = self.aux_findExtremaOnOneDOGImg(octave[s-1][0],octave[s][0],octave[s+1][0],C_DOG,C_edge)
                tmpList.append(listOfExtrema) #(NOTE) Haven't add sigma at specific scale (using interpolation)
            listOfKeypoints.append(tmpList)
        return listOfKeypoints
    
    def calGradientMagAndOrient_forPyramidL(self):
        #self.listOfOctaves
        '''self.m = np.zeros((self.numsOfOctaves,self.numsOfScales,height,width))
        self.theta = np.zeros((self.numsOfOctaves,self.numsOfScales,height,width))'''
        #self.m = self.listOfOctaves.copy()
        #self.theta = self.listOfOctaves.copy()
        self.m = np.empty((self.numsOfOctaves,0)).tolist()
        self.theta = np.empty((self.numsOfOctaves,0)).tolist()
        for o in range(len(self.listOfOctaves)): # o is octave, s is scale
            for s in range(len(self.listOfOctaves[o])): 
                #self.listOfOctaves[o][s] = (self.listOfOctaves[o][s][0].astype(np.float32),self.listOfOctaves[o][s][1])
                #L = self.listOfOctaves[o][s][0] # 0 for image, 1 for sigma
                L = self.listOfOctaves[o][s][0].astype(np.float32).copy()
                #self.m[o][s] = L.copy()
                #self.theta[o][s] = L.copy()
                height,width = L.shape
                self.m[o].append(np.zeros((height,width),np.float32))
                self.theta[o].append(np.zeros((height,width),np.float32))
                for i in range(height):
                    for j in range(width):
                        if i+1<height and j+1<width and i-1>=0 and j-1>=0:
                            self.m[o][s][i][j] = np.sqrt(pow(L[i][j+1]-L[i][j-1],2) + pow(L[i+1][j]-L[i-1][j],2)) 
                            self.theta[o][s][i][j] = np.arctan2(L[i+1][j]-L[i-1][j],L[i][j+1]-L[i][j-1])*180/np.pi
                            if self.theta[o][s][i][j] < 0:
                                self.theta[o][s][i][j] = 360 + self.theta[o][s][i][j]
         
    def aux_scaleDirections(self,thetas,numsOfDirection=8):
        h,w = thetas.shape
        new_thetas = np.copy(thetas)
        new_thetas = new_thetas
        for i in h:
            for j in w:
                if new_thetas[i,j] >= 157.5 or new_thetas[i,j] <= -157.5:
                    new_thetas[i,j] = 180
                elif -22.5 <= new_thetas[i,j] <22.5:
                    new_thetas[i,j] = 0
                elif 67.5 <= new_thetas[i,j] <= 112.5:
                    new_thetas[i,j] = 90
                elif -112.5 <= new_thetas[i,j] <= -67.5:
                    new_thetas[i,j] = 270
                elif 22.5 < new_thetas[i,j] < 67.5:
                    new_thetas[i,j] = 45
                elif 112.5 < new_thetas[i,j] < 157.5:
                    new_thetas[i,j] = 135
                elif -157.5 < new_thetas[i,j] < -112.5:
                    new_thetas[i,j] = 225
                elif -67.5 < new_thetas[i,j] < -22.5:
                    new_thetas[i,j] = 315
        return new_thetas
                
    def aux_countingBins(self,roi_m,roi_theta):
        bins_counter = np.zeros(8)
        roi_h, roi_w = roi_m.shape
        for i in roi_h:
           for j in roi_w:
               if roi_theta[i,j] == 0:
                   bins_counter[0] += roi_m[i,j]
               elif roi_theta[i,j] == 45:
                   bins_counter[1] += roi_m[i,j]
               elif roi_theta[i,j] == 90:
                   bins_counter[2] += roi_m[i,j]
               elif roi_theta[i,j] == 135:
                   bins_counter[3] += roi_m[i,j]
               elif roi_theta[i,j] == 180:
                   bins_counter[4] += roi_m[i,j]
               elif roi_theta[i,j] == 225:
                   bins_counter[5] += roi_m[i,j]
               elif roi_theta[i,j] == 270:
                   bins_counter[6] += roi_m[i,j]
               elif roi_theta[i,j] == 315:
                   bins_counter[7] += roi_m[i,j]
        return bins_counter
    
    def aux_countingBins2(self,roi_m,roi_theta,numsOfBins=36): # for 36 bins 
        lenOfBin = 360/numsOfBins
        bins_counter = np.zeros(numsOfBins)
        roi_h, roi_w = roi_m.shape
        for i in range(roi_h):
            for j in range(roi_w):
                # convert negative angle to positive
                theta = roi_theta[i,j]
                if roi_theta[i,j] < 0:
                    theta = 360 + roi_theta[i,j]
                chosenBinIndex = int(theta/lenOfBin)
                bins_counter[chosenBinIndex] += roi_m[i,j]
        return bins_counter
    
    def aux_makeFeatureVec(self,roi_m,roi_theta): # Calculate descriptor for one keypoint
        feature_vector = np.zeros(128)
        tmpIndex = 0
        for i in range(0,16,4):
            for j in range(0,16,4):
                subregion_m = roi_m[i:i+4,j:j+4]
                subregion_theta = roi_theta[i:i+4,j:j+4]
                bins_counter = self.aux_countingBins2(subregion_m,subregion_theta,8)
                
                #maxIndex = np.where(bins_counter == np.amax(bins_counter)) # get index of max value
                
                # find the bin that minimizes |angle of dominant - angle of bin|
                '''chosenBin = 0
                for b in range(8):
                    if np.abs(dominant - b*45) < chosenBin*45:
                        chosenBin = b'''
                    
                # Rotating the dominant vector to the north. 
                # Here, left-shifting all the bins so that the bin with maximum value is at index 0
                '''bins_counter = np.roll(bins_counter,8-chosenBin)'''
                # NOTE: Need more code here
                # multiple
                # assign to feature vector
                feature_vector[tmpIndex:tmpIndex+8] = bins_counter
                tmpIndex += 8
        normalized_feature_vector = normalize(feature_vector[:,np.newaxis], axis=0).ravel()
        return normalized_feature_vector
    
    def keypointDescriptor(self,k_y,k_x,mImg,thetaImg,dominant):
        sample_mKernel = np.zeros((16,16))
        sample_thetaKernel = np.zeros((16,16))
        height,width = mImg.shape
        for y in range(16):
            for x in range(16):
                # Rotate point with dominant angle
                # New x and y in kernel
                x_comma = x*np.cos(-dominant*np.pi/180) - y*np.sin(-dominant*np.pi/180)
                y_comma = x*np.sin(-dominant*np.pi/180) + y*np.cos(-dominant*np.pi/180)
                # Nearest neighbor interpolation: rounding coordinate
                x_comma = int(math.modf(x_comma)[0])
                y_comma = int(math.modf(y_comma)[0])
                # New x and y in mImg and thetaImg
                n_x = k_x+(x_comma-7)
                n_y = k_y+(y_comma-7)
                # assign corresponding value to sample kernel
                if 0<=n_x<width and 0<=n_y<height:
                    sample_mKernel[y,x] = mImg[n_y,n_x]
                    sample_thetaKernel[y,x] = thetaImg[n_y,n_x] - dominant
                    if sample_thetaKernel[y,x] < 0:
                        sample_thetaKernel[y,x] += 360
                # else: Is is necessary delete this keypoint
       # Make 16 8-bin histograms
        gaussianKernel_1D = cv2.getGaussianKernel(16,0.5*16) # sigma = 0.5*window size
        gaussianKernel_2D = np.dot(gaussianKernel_1D,gaussianKernel_1D.T)
        sample_mKernel = np.multiply(sample_mKernel,gaussianKernel_2D)
        feature_vector = self.aux_makeFeatureVec(sample_mKernel,sample_thetaKernel)
        
        return feature_vector
    
    # Find all magnitude and orientation of a certain keypoint
    def aux_find_peaks(self, hist_ori, NUM_BINS):
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
        
        return mag, ori
    
    def orientationAssignmentAndKeypointDescription(self):
        listOfKeypoints = self.findApproxKeypoints()
        # newListOfKeypoints = np.empty((self.numsOfOctaves,0)).tolist()
        self.calGradientMagAndOrient_forPyramidL()
        # traverse list of keypoints in DOG images that is not auxiliary images(start and end)
        
        for o in range(len(listOfKeypoints)): # o is octave, s is scale
            for s in range(len(listOfKeypoints[o])):
               #height,width = listOfKeypoints[o][s].shape
               
               height,width = self.listOfOctaves[o][s][0].shape
               deletedPointIndices = []
               #tmpList = []
               for k in range(len(listOfKeypoints[o][s])): # --MODDED-- due to not having storage sigma in keypoint by using interpolation # --REAL-- for keypoint in s, keypoint = (x,y,sigma)
                   # find index of image L in octave o having nearest sigma to current keypoints
                   chosenL =  s # choose the below image #--MODDED--
                   sigmaOfL = self.listOfOctaves[o][chosenL][1] #--MODDED--
                   chosen_m = self.m[o][chosenL] # m and theta has image size (type: numpy array)
                   chosen_theta = self.theta[o][chosenL]
                 
                   # keypoint coordinate
                   (k_y,k_x) = listOfKeypoints[o][s][k]
                   
                   # Now, create orientation histogram
                   # Get 16x16 samples around this keypoint
                   if k_y-7>=0 and k_y+9<height and k_x-7>=0 and k_x+9<width:
                       # Get region of interest around current keypoint
                       roi_m = chosen_m[k_y-7:k_y+9, k_x-7:k_x+9]
                       roi_theta = chosen_theta[k_y-7:k_y+9, k_x-7:k_x+9]
                       
                       #==================STEP 3: ORIENTATION ASSIGNMENT===================================
                       # (small task 1) roi_m is weighted by Gaussian kernel
                       gaussianKernel_1D = cv2.getGaussianKernel(16,1.5*sigmaOfL) #--MODDED-- sigma = 1.5*scale of the keypoint
                       gaussianKernel_2D = np.dot(gaussianKernel_1D,gaussianKernel_1D.T)
                       roi_m = np.multiply(roi_m,gaussianKernel_2D)
                       # count 36 bins histogram
                       bins_counter = self.aux_countingBins2(roi_m,roi_theta)
                       maxIndex = np.where(bins_counter == np.amax(bins_counter))[0][0] # get index of max value
                       # choose DOMINANT DIRECTION: (maxIndex+1)*10-5 --MODDED--
                       # real: fit parabola to 3 nearest bin
                       dominant = (maxIndex+1)*10 - 5
                       
                       #================STEP 4: KEYPOINT DESCRIPTOR=======================================
                       feature_vector = self.keypointDescriptor(k_y,k_x,chosen_m,chosen_theta,dominant)
                       listOfKeypoints[o][s][k] = (k_y,k_x,sigmaOfL,dominant,feature_vector)
                       #tmpList.append((k_y,k_x,dominant,feature_vector))
                   else: # delete the keypoint
                       deletedPointIndices.append(k)
               
               for index in sorted(deletedPointIndices,reverse=True):
                  del listOfKeypoints[o][s][index]
        
        return listOfKeypoints
    
    def matchTwoImages(self,listOfKeypoints1,listOfKeypoints2):
        listOfMatchingPair = []
        for (i1,k1) in enumerate(listOfKeypoints1): 
            nearest_index = -1
            nearest_dist = 9999999
            second_nearest_dist = 9999999
            for (i2,k2) in enumerate(listOfKeypoints2):
                f_vec1 = k1[4]
                f_vec2 = k2[4]
                dist = np.linalg.norm(f_vec1-f_vec2)
                if dist < nearest_dist:
                    second_nearest_dist = nearest_dist
                    nearest_index = i2
                    nearest_dist = k2
            if 0.25 <= nearest_dist/second_nearest_dist <= 0.7:
                listOfMatchingPair.append((i1,nearest_index))
        return listOfMatchingPair
    
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
                
                
       # 0, 1.25, 2.5, 3.75, 5, 6.25, 7.5, 8.75, 10 
       # lst = [[[1,3,2],[4,3],[1],[3,2]],[[3],[3,0,3,3]],[[1],[2],[3]]]                 
       # [k_y-7:k_y+9, k_x-7:k_x+9]