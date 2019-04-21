# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 17:12:04 2019

@author: VÕQUỐCHUY
"""

import numpy as np
import cv2
import sys
import convolution
import Harris
import SIFT

img = cv2.imread("./Input/flowers.jpeg", 0)
siftDetector = SIFT.SIFT(img,3,0.707,np.sqrt(2),5,4)
listOfOctaves = siftDetector.createImgs_AtMultipleOctaves()
#listOfDoG = siftDetector.calDOG_AtMultipleOctaves()
#listOfKeypoints = siftDetector.findApproxKeypoints()

listOfKeypoints = siftDetector.orientationAssignmentAndKeypointDescription()


#cv2.namedWindow('output',cv2.WINDOW_AUTOSIZE)
#cv2.imshow('output',img)

'''for i in range(len(listOfOctaves)):
    for j in range(len(listOfOctaves[i])):
        cv2.namedWindow('Output at octave'+str(i)+' scale '+str(j),cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Output at octave'+str(i)+' scale '+str(j),listOfOctaves[i][j][0])'''

'''for i in range(len(listOfDoG)):
    for j in range(len(listOfDoG[i])):
        cv2.namedWindow('Output at octave'+str(i)+' scale '+str(j),cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Output at octave'+str(i)+' scale '+str(j),listOfDoG[i][j][0])'''

for i in range(len(listOfKeypoints)):
    for j in range(len(listOfKeypoints[i])):
        print(listOfKeypoints[i][j])
        print()
        
cv2.waitKey(0)
cv2.destroyAllWindows()
