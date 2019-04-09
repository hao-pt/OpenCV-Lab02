import numpy as np
import cv2

def readImage(path):
    #Read image
    return cv2.imread(path)
def grayScale(img):
    #Convert image to grayscale
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
def writeImage(winName, img):
    #Display image
    cv2.namedWindow(winName, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(winName, img)

