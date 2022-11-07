import util.align_util as aling
import numpy as np
import math
from util.polygon import Poly
from util.torch_util import expand_dim
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import cv2 as cv
from random import random,randint
from stitching.stitch import extendAxis
from util.align_util import calcHomography, lineLineIntersection
# import util.align_util as align
from geotnf.transformation import HomographyGridGen,flex_grid_sample
from geotnf.stitching_grid_gen import StitchingHomographyGridGen
import util.align_util as aling
import util.blend_util as blend
from stitching.stitch import ImageStitcher,occlusion_prep_method
# HMatrix = [ [1.1510,  0.1892, -0.2976], [-0.0102,  0.9547, -0.0020], [-0.1242,  0.0034,
#           1.0000]]

out_w=240
out_h=240

def main():

    aux_img_src = cv.imread('ferret.jpg')    

    # img = cv.resize(aux_img_src,(240,240))
    
    # img_2 = np.zeros((480,480,3),dtype=int)

    # img_2[120:360,120:360] = img

    # cv.imwrite('img_2.jpg',img_2)

    # img_t2 = torch.FloatTensor([img_2])
    # img_t2 = Variable(img_t2.transpose(2,3).transpose(1,2)).cuda()

    # img_t = torch.FloatTensor([img])
    # img_t = Variable(img_t.transpose(2,3).transpose(1,2)).cuda()
    

    # t= torch.FloatTensor(np.array([[ 1.1510,  0.1892, -0.2976,-0.0102,  0.9547, -0.0020, -0.1242,  0.0034,1.0000]]))
    # aux_theta = Variable(t).cuda()
   
    # gen_1 = HomographyGridGen(out_h=240,out_w=240)
    # gen_2 = StitchingHomographyGridGen(img_h=240,img_w=240,out_h=480,out_w=480)

    # grid_1 = gen_1(aux_theta)
    # grid_1_flex = grid_1[0].cpu().numpy()
    # grid_2 = gen_2(aux_theta)
    # grid_2_flex = grid_2[0].cpu().numpy()

    # grid_aux = grid_2[:,120:360,120:360]
    # grid_aux_flex = grid_aux[0].cpu().numpy()

    # warped_1 = F.grid_sample(img_t,grid_1,align_corners=True)
    # warped_1_flex = flex_grid_sample(cv.transpose(img),grid_1_flex,real_h=240,real_w=240)


    # warped_2 = F.grid_sample(img_t2,grid_2,align_corners=True)
    # warped_2_flex = flex_grid_sample(cv.transpose(img),grid_2_flex,real_h=240,real_w=240)
    # warped_aux = F.grid_sample(img_t,grid_aux,align_corners=True)
    # warped_aux_flex = flex_grid_sample(cv.transpose(img),grid_aux_flex,real_h=240,real_w=240)


    # warped_1 = warped_1.transpose(1,2).transpose(2,3)[0].cpu().numpy()
    # warped_2 = warped_2.transpose(1,2).transpose(2,3)[0].cpu().numpy()
    # warped_aux = warped_aux.transpose(1,2).transpose(2,3)[0].cpu().numpy()

    # cv.imwrite('warped_1.jpg',warped_1)
    # cv.imwrite('warped_1_flex.jpg',warped_1_flex)
    # cv.imwrite('warped_2.jpg',warped_2)
    # cv.imwrite('warped_2_flex.jpg',warped_2_flex)
    # cv.imwrite('warped_aux.jpg',warped_aux)
    # cv.imwrite('warped_aux_flex.jpg',warped_aux_flex)
    # cv.imshow('warped_1',warped_1)
    # cv.waitKey(0) 

    # dif_grid = grid_1-grid_aux

    # ##############
    # M = aux_theta.reshape((-1,3))

    # # Generating src img with random height, width and position

    # # stitcher = ImageStitcher('hom','resnet101')

    # grid_gen = StitchingHomographyGridGen()

    # grid = grid_gen(aux_theta)
    
    # # grid = grid[0].detach().cpu().numpy()
    
    # weights = blend.genWieghts(out_w, out_h)

    # srcImg = fitForBlending(img)
    # weights = fitForBlending(weights)
    
    # srcImg=torch.FloatTensor([srcImg])
    # weights =torch.FloatTensor([weights])


    # srcImg = Variable(srcImg,requires_grad=False).cuda()
    # srcImg = srcImg.transpose(2,3).transpose(1,2)

    # weights = Variable(weights,requires_grad=False).cuda()
    # weights = weights.transpose(2,3).transpose(1,2)

    # auxImg = F.grid_sample(srcImg, grid,align_corners=True)
    # auxWeights = F.grid_sample(weights, grid,align_corners=True)

    # auxImg = auxImg[0]
    # auxImg = auxImg.transpose(0,1).transpose(1,2).detach().cpu().numpy()

    # cv.imwrite("gridTest.jpg",auxImg)

    # print(grid)

    # flex_grid_sample(img,grid,out_h=480,out_w=480)

    imgList = [ "/workspaces/GeometricCNN/apifiles/uncompressed/0e4407b0-44cc-11ed-a64d-b3a88c7b2dcc/foto_1.jpg",
                "/workspaces/GeometricCNN/apifiles/uncompressed/0e4407b0-44cc-11ed-a64d-b3a88c7b2dcc/foto_2.jpg",
                "/workspaces/GeometricCNN/apifiles/uncompressed/0e4407b0-44cc-11ed-a64d-b3a88c7b2dcc/foto_3.jpg",
                "/workspaces/GeometricCNN/apifiles/uncompressed/0e4407b0-44cc-11ed-a64d-b3a88c7b2dcc/foto_4.jpg",
                "/workspaces/GeometricCNN/apifiles/uncompressed/0e4407b0-44cc-11ed-a64d-b3a88c7b2dcc/foto_5.jpg",
                "/workspaces/GeometricCNN/apifiles/uncompressed/0e4407b0-44cc-11ed-a64d-b3a88c7b2dcc/foto_6.jpg"  ]
    

    # stitcher =  ImageStitcher('hom', 'vgg','occlusion')
    # stitcher =  ImageStitcher('aff', 'resnet101','occlusion')
    stitcher =  ImageStitcher('tps', 'resnet101','occlusion')


    stitcher.stitch(imgList,"/workspaces/GeometricCNN/cache")

    
    # test_src,test_tgt = occlusion_prep_method(aux_img_src,aux_img_src,2)

    # cv.imwrite("occluded_src.jpg",test_src)
    # cv.imwrite("occluded_tgt.jpg",test_tgt)


if __name__ == "__main__":

    # N = 100000
    # a = cuda.to_device(np.random.random(N))
    # b = cuda.to_device(np.random.random(N))
    # c = cuda.device_array_like(a)

    # print(c.flags)
    # f.forall(len(a))(a, b, c)
    # print(c.copy_to_host())
    
    main()