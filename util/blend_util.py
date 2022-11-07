import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))
print(str(Path.cwd()))

import math
from typing import List
import numpy as np
import scipy.ndimage as ndimage
from sklearn.preprocessing import normalize
from util.align_util import NormalizeData

def genWieghts(width,height):
    
    w_map = np.zeros((height,width,3))

    w_map[:height,0] = [1,1,1]
    w_map[:height,-1] = [1,1,1]
    w_map[0,:width] = [1,1,1]
    w_map[-1,:width] = [1,1,1]

    # print(w_map[:height,0])

    w_map = ndimage.distance_transform_edt(1-w_map)

    return NormalizeData(w_map)

def blendImgs(img_stack,w_stack,outDim):

    newImg = np.zeros((img_stack.shape[1:]))

    for j in (range(img_stack.shape[1])):

        for i in (range(img_stack.shape[2])):
            
            newPixel = np.zeros((3))
            weight = np.zeros((3))
            
            stackedNum = 0
            lastNonZero = -1

            for k in (range(img_stack.shape[0])):
                
                if (w_stack[k,j,i,0]!=0):
                    newPixel= newPixel + img_stack[k,j,i] * w_stack[k,j,i]
                    weight= weight + w_stack[k,j,i]
                    stackedNum= stackedNum + 1
                    lastNonZero = k

            if (weight[0]!=0):
                newPixel/=weight
            
            if stackedNum>1:
                newImg[j,i] = newPixel
            else:
                newImg[j,i] = img_stack[lastNonZero,j,i]

    y_start = outDim[0]//2
    y_end = newImg.shape[0]-y_start

    x_start = outDim[1]//2
    x_end = newImg.shape[1]-x_start

    newImg = newImg[y_start:y_end,x_start:x_end]

    return newImg


def calcNewSpace(imgShape: np.array, numImg: int) -> np.array:

    numStrips = math.ceil(numImg / 3) 

    height = imgShape[0] * 3 
    width = ((imgShape[1]//2) * (numStrips+1)) + imgShape[1]

    return np.array( [height,width,imgShape[-1]] )


#Fits the image on the new space assuming thirds for height and width that difers by half of the out dimensions
# The coords refers to the imaginary cuadrants on the space 
def fitInSpace(img: np.array, spaceShape: np.array, outDimensions: List, coords: List) -> np.array:

    y_start = coords[1]*(outDimensions[0]//2)
    y_end = y_start+(img.shape[0])

    x_start = coords[0]*(outDimensions[1]//2)
    x_end = x_start+(img.shape[1])

    newImg = np.zeros(spaceShape)

    newImg[y_start:y_end,x_start:x_end] = img

    return newImg