#Built-in libs
import numpy as np
import cv2
import argparse
import math
from matplotlib import pyplot as plt
import argparse

#Built-out libs
import myImage
import filter as flt
import harris
import mysift
import blob


#Instatiate ArgumentParser() obj and parse argument
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required = True, help = "Path to input image")
ap.add_argument("-c", "--code", required= True, help = "Code action")
args = vars(ap.parse_args())

def main(args):
    #Measure time
    e1 = cv2.getTickCount()

    # input gray image
    grayImg = myImage.readImage(args["input"])
    
    # Convert img in range[0,1]
    floatImg = grayImg / 255.0

    # Get code action
    code = int(args['code'])

    if code == 1:
        # Call harris detector function
        harris.detectByHarris(floatImg, _ratio = 0.1)
    elif code == 2:
        # Detect Blob used LoG
        # Declare CBlob obj
        myblob = blob.CBlob()
        myblob.detectBlobByLoG(floatImg)
        myblob.plotBlob(floatImg)
    elif code == 3:
        # Detect Blob used DoG
        # Declare sift obj
        mySift = mysift.CSift(floatImg)
        mySift.detectBlobByDoG()
        mySift.plotBlob()
    elif code == 4:
        # Sift detector
        # Declare sift obj
        mySift = mysift.CSift(floatImg)
        mySift.detectBySift()

    e2 = cv2.getTickCount()
    time = (e2 - e1)/cv2.getTickFrequency()
    print('Time: %.2f(s)' %(time))

    plt.show()

    return 1


if __name__ == "__main__":
    main(args)