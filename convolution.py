import numpy as np
import cv2
from skimage import exposure, img_as_ubyte

class CMyConvolution:
    #Ham khoi tao
    def __init__(self):
        #Height & width of kernel
        self.kH = self.kW = 0
        #Init kernel by 3x3 matrix with 0-element
        self.kernel = np.zeros((3, 3))

    #Gan kernel voi 1 mask cho truoc
    def setKernel(self, mask):
        self.kH, self.kW = mask.shape
        self.kernel = mask
    
    #Flip filter
    def flipFilter(self):
        return self.kernel[::-1, ::-1]

    #Convolution of img and kernel/mask
    def convolution(self, img): 
        #Padding de output image keep the same size as input image
        #Tinh padding them vao
        pV = (self.kH - 1) // 2
        pH = (self.kW - 1) // 2
        #Padding
        paddingImg = cv2.copyMakeBorder(img, pV, pV, pH, pH, cv2.BORDER_REPLICATE)

        #Size of img
        iH, iW = img.shape

        # Output image
        outImg = np.zeros(img.shape, dtype=np.float64)

        #flip filter then multiply element-wise
        flipKernel = self.flipFilter()

        #Scan image without padding indices. Cause center of kernel slide in each pixel of image
        for y in range(pV, iH):
            for x in range(pH, iW):
                #Extract the roi (Region of interesting) that have same size as kernel
                roi = paddingImg[y - pV: y + pV + 1, x-pH: x + pH + 1]

                #Element-wise multiplication of roi & kernel then get the sum of it 
                # to get the convolve output
                k = abs((roi * flipKernel).sum())

                #Assign this convole output to pixel (y, x) of output image
                #Note: Ouput size remain the same as the original img
                #Use method: .itemset of numpy to speed up modify pixel in image 
                outImg.itemset((y - pV, x - pH), k)
        
        # # Normalize the output image to be in range [0, 255] accurately_
        # # _when it's presented in float dtype [0, 1] called 'shrinking image'
        outImg = exposure.rescale_intensity(outImg, out_range = (0, 1))
        # # Convert dtype of image back to uint8
        outImg = (outImg * 255).astype(np.uint8)
        return outImg