# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 16:34:17 2019

@author: VÕQUỐCHUY
"""
import numpy as np
import cv2
from numpy import linalg as LA

class SIFT:
    def __init__(self,_img,_kernelSize,_sigma,_k,_numsOfScales,_numsOfOctaves):
        self.img = _img
        self.kernelSize = _kernelSize
        self.sigma = _sigma
        self.k = _k
        self.numsOfScales = _numsOfScales
        self.numsOfOctaves = _numsOfOctaves
        
    def createImgs_AtOctave(self,img):
        height,width = img.shape
        listOfImgs = []
        for i in range(self.numsOfScales):
            blurImg = cv2.GaussianBlur(img,(self.kernelSize,self.kernelSize),self.sigma*pow(self.k,i))
            listOfImgs.append((blurImg,self.sigma*pow(self.k,i)))
        return listOfImgs
    
    def createImgs_AtMultipleOctaves(self):
        # not complete: choose the start sigma for the next octave
        listOfOctaves = []
        sigma = self.sigma
        for i in range(self.numsOfOctaves):
            resizedImg = cv2.resize(self.img, (0,0), fx=pow(0.5,i), fy=pow(0.5,i))
            octave = self.createImgs_AtOctave(resizedImg)
            listOfOctaves.append(octave)
            sigma *= pow(self.k,(self.numsOfScales//2))
        return listOfOctaves
    
    def calDOG_AtMultipleOctaves(self):
        listOfOctaves = self.createImgs_AtMultipleOctaves()
        listOfDOGImgs = []
        for octave in listOfOctaves:
            tmpList = []
            for i in range(1,len(octave)):
                tmpImg1 = octave[i][0].astype(np.int8)
                tmpImg2 = octave[i-1][0].astype(np.int8)
                DOGImgs = tmpImg1 - tmpImg2
                DOGImgs[DOGImgs<0] = abs(DOGImgs[DOGImgs<0]) +50
                DOGImgs = DOGImgs.astype(np.uint8)
                tmpList.append((DOGImgs,octave[i-1][1])) # ?? i-1 or i
            listOfDOGImgs.append(tmpList)
        return listOfDOGImgs
    
    def aux_findExtremaOnOneDOGImg(self,img_below,img_middle,img_above,C_DOG,C_edge):# return a list of tuples
        height,width = img_middle.shape
        eight_neighbors = np.array([[0,-1,-1,0,1,1,1,0,-1],
							          [0,0,1,1,1,0,-1,-1,-1]], np.int)
        returnList = []
        for i in range(height):
            for j in range(width):
                # get 27 neighbor pixels
                listOfNeighborVals = []
                for k in range(8):
                    dy,dx = eight_neighbors[0][k],eight_neighbors[1][k]
                    ny = i + dy
                    nx = j + dx
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
                        h11 = int_img_middle[j,i+1] + int_img_middle[j,i-1] - 2*int_img_middle[j,i]
                        h22 = int_img_middle[j+1,i] + int_img_middle[j-1,i] - 2*int_img_middle[j,i]
                        h12 = (int_img_middle[j+1,i+1] - int_img_middle[j-1,i+1] - int_img_middle[j+1,i-1] + int_img_middle[j-1,i-1])/4
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
                tmpList.append(listOfExtrema) #(NOTE) Haven't add sigma at specific scale
            listOfKeypoints.append(tmpList)
        return listOfKeypoints
    
    
                        
                