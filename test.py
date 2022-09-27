import util.align_util as aling
import numpy as np
import math
from util.polygon import Poly
from util.torch_util import expand_dim
import torch
from torch.autograd import Variable
import cv2 as cv
from random import random,randint
from stitching.stitch import extendAxis
from util.align_util import calcHomography, lineLineIntersection
# import util.align_util as align
from numba import cuda
# HMatrix = [ [1.1510,  0.1892, -0.2976], [-0.0102,  0.9547, -0.0020], [-0.1242,  0.0034,
#           1.0000]]


def main():

    aux_img_src = cv.imread('ferret.jpg')
    aux_theta = np.array([ 1.1510,  0.1892, -0.2976,-0.0102,  0.9547, -0.0020, -0.1242,  0.0034,1.0000])
   
    ##############
    M = aux_theta.reshape((-1,3))

    # Generating src img with random height, width and position

    w_random = 0.25+(random()*(0.5-0.25))
    h_random = 0.25+(random()*(0.5-0.25))

    img_w = math.floor(aux_img_src.shape[1]*w_random)
    img_h = math.floor(aux_img_src.shape[0]*h_random)

    x_random = 0.25+(random()*(0.5-0.25))
    y_random = 0.25+(random()*(0.5-0.25))

    x_offset = math.floor(aux_img_src.shape[1]*x_random)
    y_offset = math.floor(aux_img_src.shape[0]*y_random)

    # Extracting src img
    img_src = aux_img_src[y_offset:y_offset+img_h,x_offset:x_offset+img_w]
    # cv.imwrite("src.jpg",img_src)
    img_src = cv.resize(img_src,(240,240))

    # cropped_image_batch[i] = torch.Tensor(img_src.astype(np.float32)).transpose(1,2).transpose(0,1)
    # # cropped_image_batch[i] = cropped_image_batch[i].transpose(1,2).transpose(0,1)
    # cropped_image_batch[i] = Variable(cropped_image_batch[i],requires_grad=False)

    # Generating trg img
        
    # b, c, h, w = image_batch.size()

    # generate symmetrically padded image for bigger sampling region
    # image_batch = self.symmetricImagePad(image_batch,self.padding_factor)
    # 
    aux_img_trg = np.pad(aux_img_src,((int(aux_img_src.shape[0]*0.5),int(aux_img_src.shape[0]*0.5)),
                                        (int(aux_img_src.shape[1]*0.5),int(aux_img_src.shape[1]*0.5)),
                                        (0,0)),'symmetric')
    
    # #Generate normalized space to work with homography
    src_X_axis = np.linspace(-1,1,aux_img_src.shape[1])
    src_Y_axis = np.linspace(-1,1,aux_img_src.shape[0])
    
    # Generate normalized space for padded image
    X_axis = extendAxis(src_X_axis)
    Y_axis = extendAxis(src_Y_axis)
    
    #generate normalized homography space
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

    #Define boundbox of padded image to apply homography
    boundBox = [ [X_axis[0],Y_axis[-1]],
                    [X_axis[-1],Y_axis[-1]],
                    [X_axis[-1],Y_axis[0]],
                    [X_axis[0],Y_axis[0]],
                        ]
    boundBox = [ calcHomography(M,x) for x in boundBox ]

    #Filtering points
    h_points = aling.getImgPoints(boundBox,points)


    #Apply homography to img
    aux_img_trg_H=aling.applyHom(aux_img_trg,h_points,H_X_axis,np.flip(H_Y_axis),X_axis,np.flip(Y_axis),M)


    # cv.imwrite("aux_trg.jpg",aux_img_trg_H)


    #Flip Y axis to denormalize points
    Y_axis = np.flip(Y_axis)
    H_Y_axis = np.flip(H_Y_axis)

    #Define boundbox of the src img on the hom space
    src_corners = [ [src_X_axis[x_offset],src_Y_axis[y_offset]],
                    [src_X_axis[x_offset+img_w],src_Y_axis[y_offset]],
                    [src_X_axis[x_offset+img_w],src_Y_axis[y_offset+img_h]],
                    [src_X_axis[x_offset],src_Y_axis[y_offset+img_h]],
                        ]

    boundBox = [ calcHomography(M,x) for x in src_corners ]
    
    # Calc starting point to extract trg img
    center = lineLineIntersection(boundBox[0],boundBox[2],boundBox[3],boundBox[1])
    center_coords = [0,0] 
    center_coords[0] = min(range(len(H_X_axis)), key=lambda i: abs(H_X_axis[i]-center[0]))
    center_coords[1] = min(range(len(H_Y_axis)), key=lambda i: abs(H_Y_axis[i]-center[1]))

    #Random generate directions to move the trg
    directions=[-1,1]
    x_direction = directions[ randint(0,1) ]
    y_direction = directions[ randint(0,1) ]

    x_offset = randint( 0 , int(img_w*0.15))

    y_offset = randint( 0 , int(img_h*0.15))

    x_offset = x_offset*x_direction
    y_offset = y_offset*y_direction

    #Moving center
    center_coords[0] = center_coords[0] + x_offset
    center_coords[1] = center_coords[1] + y_offset
    
    # Randomly take reference to move trg img
    ref_point = randint(0,3)

    # Randomly take side or corner as reference point
    side = randint(0,1) 

    #Prepare start_point to extract trg img
    if ref_point == 0:
        start_point = center_coords

        start_point[0] = start_point[0]-int(side*img_w*0.5)

    elif ref_point == 1 :
        start_point = [ center_coords[0]-img_w,center_coords[1] ]

        start_point[1] = start_point[1] - int(side*img_h*0.5)

    elif ref_point == 2 :
        start_point = [ center_coords[0]-img_w,center_coords[1] - img_h ]

        start_point[0] = start_point[0] + int(side*img_w*0.5)

    elif ref_point == 3 :
        start_point = [ center_coords[0],center_coords[1] - img_h ]

        start_point[1] = start_point[1] + int(side*img_h*0.5)


    #Extract trg img
    trg_img = aux_img_trg_H[start_point[1]:start_point[1]+img_h,start_point[0]:start_point[0]+img_w]
    # cv.imwrite("trg.jpg",trg_img)
    trg_img = cv.resize(trg_img,(240,240))



# @cuda.jit
# def f(a, b, c):
#     # like threadIdx.x + (blockIdx.x * blockDim.x)
#     tid = cuda.grid(1)
#     size = len(c)

#     if tid < size:
#         c[tid] = a[tid] + b[tid]


if __name__ == "__main__":

    # N = 100000
    # a = cuda.to_device(np.random.random(N))
    # b = cuda.to_device(np.random.random(N))
    # c = cuda.device_array_like(a)

    # print(c.flags)
    # f.forall(len(a))(a, b, c)
    # print(c.copy_to_host())
    
    main()