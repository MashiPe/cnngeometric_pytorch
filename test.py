import util.align_util as aling
import numpy as np
import math
from util.torch_util import expand_dim
import torch
from torch.autograd import Variable
import cv2 as cv
# HMatrix = [ [1.1510,  0.1892, -0.2976], [-0.0102,  0.9547, -0.0020], [-0.1242,  0.0034,
#           1.0000]]

def extendAxis(axis):

    start = axis[0]
    end = axis[-1]

    dist = end-start
    step = dist / (len(axis)-1)

    left_ext = np.arange(start=start,stop= (start-int(dist/2)),step=-step )
    left_ext = left_ext[1:]
    left_ext = np.append(left_ext,math.floor(left_ext[-1]))
    left_ext = np.flip(left_ext)

    right_ext = np.arange(start=end,stop= (end+int(dist/2)),step=step )
    right_ext = right_ext[1:]
    right_ext = np.append(right_ext,math.ceil(right_ext[-1]))

    return np.concatenate([left_ext,axis,right_ext])


def main():

    out_w = 240
    out_h = 240
    startNormSpace= -1
    endNormSpace = 1
    
    img_stack = []
    w_stack = []


    X_axis = np.linspace(startNormSpace,endNormSpace,out_w)
    Y_axis = np.linspace(startNormSpace,endNormSpace,out_h)

    H_X_axis = extendAxis(X_axis)
    H_Y_axis = extendAxis(Y_axis)

    grid_X,grid_Y = np.meshgrid(H_X_axis,H_Y_axis)

    #Process to generate all points in the space 
    grid_X = torch.DoubleTensor(grid_X).unsqueeze(0).unsqueeze(3)
    grid_Y = torch.DoubleTensor(grid_Y).unsqueeze(0).unsqueeze(3)
    grid_X = Variable(grid_X,requires_grad=False)
    grid_Y = Variable(grid_Y,requires_grad=False)
    grid_X = expand_dim(grid_X,0,1)
    grid_Y = expand_dim(grid_Y,0,1)

    grid_points = torch.cat((grid_X,grid_Y),3)

    grid_points = grid_points.numpy()[0]

    #Reshape array to get a list of pair points
    points = grid_points.reshape((-1-1,2))


    #Process image homography
    HMatrix = [ [1.1332,  0.0198, -0.4599], [0.0134,  0.8718, -0.0281], [-0.2586, -0.0053,
            1.0000]]

    boundBox = aling.maxBoundBox(HMatrix)

    h_points = aling.getImgPoints(boundBox,points)

    img = cv.imread("./datasets/testDataset/street2.jpg")
    img = cv.resize(img,(out_w,out_h))

    newImg=aling.applyHom(img,h_points,H_X_axis,np.flip(H_Y_axis),X_axis,np.flip(Y_axis),HMatrix)

    #Process image weights for blending

    weights = aling.genWieghts(out_w,out_h)

    h_weights=aling.applyHom(weights,h_points,H_X_axis,np.flip(H_Y_axis),X_axis,np.flip(Y_axis),HMatrix,singleDimension=True)

    img_stack.append(newImg)
    w_stack.append(h_weights)


    #Another img
    HMatrix = np.array( [ [1,0,-1],[0,1,0],[0,0,1] ] )

    boundBox = aling.maxBoundBox(HMatrix)

    h_points = aling.getImgPoints(boundBox,points)

    img = cv.imread("./datasets/testDataset/street1.jpg")
    img = cv.resize(img,(out_h,out_w))

    newImg=aling.applyHom(img,h_points,H_X_axis,np.flip(H_Y_axis),X_axis,np.flip(Y_axis),HMatrix,one2oneMaping=True)
    h_weights=aling.applyHom(weights,h_points,H_X_axis,np.flip(H_Y_axis),X_axis,np.flip(Y_axis),HMatrix,one2oneMaping=True,singleDimension=True)

    img_stack.append(newImg)
    w_stack.append(h_weights)

    cv.imwrite("test1.jpg",newImg)

    img_stack = np.array(img_stack)
    w_stack = np.array(w_stack)

    newImg = aling.blendImgs(img_stack,w_stack)

    cv.imwrite("testb.jpg",newImg)


    


if __name__ == "__main__":
    main()