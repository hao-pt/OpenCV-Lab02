#Built-in libs
import numpy as np
import cv2
import argparse
import math
from matplotlib import pyplot as plt

#Built-out libs
import myImage
import filter as flt
import harris
import sift

def main():
    #Measure time
    e1 = cv2.getTickCount()

    # input image
    img = myImage.readImage("empire.jpg")
    # Grayscale image
    grayImg = myImage.grayScale(img)

    plt.figure(1)
    plt.imshow(grayImg, cmap='gray', interpolation = 'bicubic')
    plt.title('Gray-scale image'), plt.xticks([]), plt.yticks([])

    # Call harris detector function
    # harris.detectByHarris(grayImg)
    
    # Declare sift obj
    mysift = sift.CSift(grayImg)
    mysift.detectBySift()
    
    e2 = cv2.getTickCount()
    time = (e2 - e1)/cv2.getTickFrequency()
    print('Time: %.2f(s)' %(time))

    plt.show()


if __name__ == "__main__":
    main()