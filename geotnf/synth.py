from __future__ import print_function, division
import os
from symbol import varargslist
import sys
from tracemalloc import start
from skimage import io
import pandas as pd
import numpy as np
import torch
from torch.nn.modules.module import Module
from torch.utils.data import Dataset
from torch.autograd import Variable
import torch.nn.functional as F
from util.torch_util import expand_dim
from util.align_util import maxBoundBox,calcHomography
from image.normalization import normalize_image
from geotnf.transformation import GeometricTnf
from geotnf.transformation import homography_mat_from_4_pts
from random import random,randint
import math
import cv2 as cv
from stitching.stitch import extendAxis


class SynthPairStitchTnf(object):
    """
    
    Generate a synthetically warped training pair using an affine transformation.
    
    """
    def __init__(self, use_cuda=True, geometric_model='affine', crop_factor=9/16, output_size=(240,240), padding_factor = 0.5, occlusion_factor=0):
        assert isinstance(use_cuda, (bool))
        assert isinstance(crop_factor, (float))
        assert isinstance(output_size, (tuple))
        assert isinstance(padding_factor, (float))
        self.occlusion_factor=occlusion_factor
        self.use_cuda=use_cuda
        self.crop_factor = crop_factor
        self.padding_factor = padding_factor
        self.out_h, self.out_w = output_size 
        self.rescalingTnf = GeometricTnf('affine', out_h=self.out_h, out_w=self.out_w, 
                                         use_cuda = self.use_cuda)
        self.geometricTnf = GeometricTnf(geometric_model, out_h=self.out_h, out_w=self.out_w, 
                                         use_cuda = self.use_cuda)

        
    def __call__(self, batch):
        images_batch, thetas_batch = batch['image'], batch['theta']

        # generate symmetrically padded image for bigger sampling region
        padded_images_batch = self.symmetricImagePad(images_batch,self.padding_factor)
        
        batch_shape = (images_batch.size(dim=0),images_batch.size(dim=1),240,240)

        cropped_image_batch = torch.empty(batch_shape)
        warped_image_batch = torch.empty(batch_shape)
        theta_image_batch = torch.empty(thetas_batch.size())
        polygons_batch = torch.empty((thetas_batch.size(dim=0),4,2))

        for i in range(images_batch.size(dim=0)):
            image_batch = images_batch[i]
            theta_batch = thetas_batch[i]            

            aux_img_src = image_batch.transpose(0,1).transpose(1,2).cpu().numpy()
            aux_theta = homography_mat_from_4_pts(Variable(theta_batch.unsqueeze(0))).squeeze(0).data
            M = aux_theta.reshape((-1,3)).numpy()

            # Generating src img 

            w_random = 0.25+(random()*(0.5-0.25))
            h_random = 0.25+(random()*(0.5-0.25))

            img_w = math.floor(aux_img_src.shape[1]*w_random)
            img_h = math.floor(aux_img_src.shape[0]*h_random)

            x_random = 0.25+(random()*(0.5-0.25))
            y_random = 0.25+(random()*(0.5-0.25))

            x_offset = math.floor(aux_img_src.shape[1]*x_random)
            y_offset = math.floor(aux_img_src.shape[0]*y_random)

            # print([y_offset,img_h,x_offset,img_w])

            img_src = aux_img_src[y_offset:y_offset+img_h,x_offset:x_offset+img_w]
            # cv.imwrite("src.jpg",img_src)
            img_src = cv.resize(img_src,(240,240))

            cropped_image_batch[i] = torch.Tensor(img_src.astype(np.float32)).transpose(1,2).transpose(0,1)
            # cropped_image_batch[i] = cropped_image_batch[i].transpose(1,2).transpose(0,1)
            cropped_image_batch[i] = Variable(cropped_image_batch[i],requires_grad=False)

            # Generating trg img
            
            if self.use_cuda:
                image_batch = image_batch.cuda()

                
            # b, c, h, w = image_batch.size()

            # generate symmetrically padded image for bigger sampling region
            # image_batch = self.symmetricImagePad(image_batch,self.padding_factor)
            aux_img_trg = padded_images_batch[i].transpose(0,1).transpose(1,2).cpu().numpy()

            aux_img_trg =  cv.warpPerspective(aux_img_trg,M,aux_img_trg.shape[:2])
            #Generate normalized space to work with homography
            X_axis = np.linspace(-1,1,aux_img_src.shape[1])
            Y_axis = np.linspace(-1,1,aux_img_src.shape[0])
            
            H_X_axis = extendAxis(X_axis)
            H_Y_axis = np.flip(extendAxis(Y_axis))

            src_corners = [ [Y_axis[y_offset],X_axis[x_offset]],
                            [Y_axis[y_offset],X_axis[x_offset+img_w]],
                            [Y_axis[y_offset+img_h],X_axis[x_offset+img_w]],
                            [Y_axis[y_offset+img_h],X_axis[x_offset]],
                             ]

            boundBox = [ calcHomography(M,x) for x in src_corners ]

            # boundBox = maxBoundBox(M)


            
            # Calc starting point to extract trg img
            center = self.lineLineIntersection(boundBox[0],boundBox[2],boundBox[3],boundBox[1])
            center_coords = [0,0] 
            center_coords[0] = min(range(len(H_X_axis)), key=lambda i: abs(H_X_axis[i]-center[0]))
            center_coords[1] = min(range(len(H_Y_axis)), key=lambda i: abs(H_Y_axis[i]-center[1]))

            #Random generate directions to move the trg
            directions=[-1,1]
            x_direction = directions[ randint(0,1) ]
            y_direction = directions[ randint(0,1) ]

            # if (x_direction == 1):
            #     x_offset = randint( int(img_w*0.5) , img_w )
            # else:
            x_offset = randint( 0 , int(img_w*0.1))

            # if (y_direction == 1):
            #     y_offset = randint( int(img_h*0.5) , img_h )
            # else:
            y_offset = randint( 0 , int(img_h*0.1))

            x_offset = x_offset*x_direction
            y_offset = y_offset*y_direction

            center_coords[0] = center_coords[0] + x_offset
            center_coords[1] = center_coords[1] + y_offset
            
            # Randomly take a corner as reference to move trg img
            ref_point = (randint(0,3))

            # start_point = [0,0]

            if ref_point == 0:
                start_point = center_coords

                point_0 = self.lineLineIntersection([0,start_point[1]],start_point,boundBox[2],boundBox[1])
                point_1 = self.lineLineIntersection([start_point[0],0],start_point,boundBox[3],boundBox[2])

                polygon = [ start_point,point_0,boundBox[2],point_1 ]

            elif ref_point == 1 :
                start_point = [ center_coords[0]-img_w,center_coords[1] ]

                point_0 = self.lineLineIntersection([0,start_point[1]],start_point,boundBox[3],boundBox[0])
                point_1 = self.lineLineIntersection([start_point[0],0],start_point,boundBox[3],boundBox[2])

                polygon = [ point_0,start_point,point_1,boundBox[3]]
            elif ref_point == 2 :
                start_point = [ center_coords[0]-img_w,center_coords[1] - img_h ]

                point_0 = self.lineLineIntersection([0,start_point[1]],start_point,boundBox[3],boundBox[0])
                point_1 = self.lineLineIntersection([start_point[0],0],start_point,boundBox[0],boundBox[1])

                polygon = [ boundBox[0],point_1,point_0,start_point]
            elif ref_point == 3 :
                start_point = [ center_coords[0],center_coords[1] - img_h ]

                point_0 = self.lineLineIntersection([0,start_point[1]],start_point,boundBox[2],boundBox[1])
                point_1 = self.lineLineIntersection([start_point[0],0],start_point,boundBox[0],boundBox[1])

                polygon = [ point_1,boundBox[1],point_0,start_point]
            

            trg_img = aux_img_trg[start_point[1]:start_point[1]+img_h,start_point[0]:start_point[0]+img_w]
            # cv.imwrite("trg.jpg",trg_img)
            trg_img = cv.resize(trg_img,(240,240))
            warped_image_batch[i] = torch.Tensor(trg_img.astype(np.float32)).transpose(1,2).transpose(0,1)
            # warped_image_batch[i] = warped_image_batch[i].transpose(1,2).transpose(0,1)
            warped_image_batch[i] = Variable(warped_image_batch[i],requires_grad=False)

            theta_image_batch[i] = theta_batch

            

            polygons_batch[i] = torch.Tensor(polygon)
            polygons_batch[i] = Variable(polygons_batch[i],requires_grad=False)


        if self.use_cuda:
            cropped_image_batch = cropped_image_batch.cuda()
            warped_image_batch = warped_image_batch.cuda()
            theta_image_batch = theta_image_batch.cuda()
        
        return {'source_image': cropped_image_batch, 'target_image': warped_image_batch, 'theta_GT': theta_image_batch,
                'polygon': polygons_batch}
        

    def symmetricImagePad(self, image_batch, padding_factor):
        b,c, h, w = image_batch.size()
        pad_h, pad_w = int(h*padding_factor), int(w*padding_factor)
        idx_pad_left = torch.LongTensor(range(pad_w-1,-1,-1))
        idx_pad_right = torch.LongTensor(range(w-1,w-pad_w-1,-1))
        idx_pad_top = torch.LongTensor(range(pad_h-1,-1,-1))
        idx_pad_bottom = torch.LongTensor(range(h-1,h-pad_h-1,-1))
        # if self.use_cuda:
        #         idx_pad_left = idx_pad_left.cuda()
        #         idx_pad_right = idx_pad_right.cuda()
        #         idx_pad_top = idx_pad_top.cuda()
        #         idx_pad_bottom = idx_pad_bottom.cuda()
        image_batch = torch.cat((image_batch.index_select(3,idx_pad_left),image_batch,
                                 image_batch.index_select(3,idx_pad_right)),3)
        image_batch = torch.cat((image_batch.index_select(2,idx_pad_top),image_batch,
                                 image_batch.index_select(2,idx_pad_bottom)),2)
        return image_batch

    def get_occlusion_mask(self, mask_size, occlusion_factor):
        b, c, out_h, out_w = mask_size
        # create mask of occluded portions
        box_w = torch.round(out_w*torch.sqrt(torch.FloatTensor([occlusion_factor]))*(1+(torch.rand(b)-0.5)*2/5))
        box_h = torch.round(out_h*out_w*occlusion_factor/box_w); 
        box_x = torch.floor(torch.rand(b)*(out_w-box_w));
        box_y = torch.floor(torch.rand(b)*(out_h-box_h));
        box_w = box_w.int()
        box_h = box_h.int()
        box_x = box_x.int()
        box_y = box_y.int()
        mask = torch.zeros(mask_size)
        for i in range(b):
            mask[i,:,box_y[i]:box_y[i]+box_h[i],box_x[i]:box_x[i]+box_w[i]]=1        
        # convert to variable
        mask = Variable(mask)
        return mask

    def lineLineIntersection(self,A, B, C, D):
        # Line AB represented as a1x + b1y = c1
        a1 = B[1] - A[1]
        b1 = A[0] - B[0]
        c1 = a1*(A[0]) + b1*(A[1])
    
        # Line CD represented as a2x + b2y = c2
        a2 = D[1] - C[1]
        b2 = C[0] - D[0]
        c2 = a2*(C[0]) + b2*(C[1])
    
        determinant = a1*b2 - a2*b1
    
        x = (b2*c1 - b1*c2)/determinant
        y = (a1*c2 - a2*c1)/determinant
        return [x, y]

    def evalLine(self,A, B, x):
        # Line AB represented as a1x + b1y = c1
        a1 = B[1] - A[1]
        b1 = A[0] - B[0]
        c1 = a1*(A[0]) + b1*(A[1])

        # Funtion as y = (-a1x + c1 )/b1
        y = ( (-a1*x)+c1 )/b1
    
        return [x, y]