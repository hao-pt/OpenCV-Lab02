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

def main():
    # input image
    img = myImage.readImage("empire.jpg")
    # Grayscale image
    grayImg = myImage.grayScale(img)

    plt.figure(1)
    plt.imshow(grayImg, cmap='gray', interpolation = 'bicubic')
    plt.title('Gray-scale image'), plt.xticks([]), plt.yticks([])

    # Call harris detector function
    harris.detectByHarris(grayImg)
    
    

    plt.show()


if __name__ == "__main__":
    main()