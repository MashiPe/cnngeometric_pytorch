from argparse import Namespace
from os import makedirs, path
import util.align_util as aling
import util.blend_util as blend
import numpy as np
import math
from util.torch_util import expand_dim
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import cv2 as cv
from model.cnn_geometric_model import CNNGeometric
from collections import OrderedDict
from geotnf.transformation import GeometricTnf
from torchvision.transforms import Normalize
from image.normalization import normalize_image
from skimage import io
import sys
from geotnf.transformation import homography_mat_from_4_pts,flex_grid_sample
from geotnf.stitching_grid_gen import StitchingTpsGridGen,StitchingHomographyGridGen,StitchingAffineGridGen
from typing import Any, List, Tuple
from multipledispatch import dispatch
from numpy.typing import ArrayLike
from stitching.PairImage import PairImage
from stitching.Image import Image
import matplotlib.pyplot as plt


out_w = 240
out_h = 240
startNormSpace= -1
endNormSpace = 1
sigma=1
num_bands=5

USE_CUDA = torch.cuda.is_available()


models_paramas = {
    # No occlusion models
    'tps_4p_no_occ':{
        'model_path': 'trained_models/tps_4p_no_occ/best_tps_4p_no_occ_tps_grid_lossresnet101.pth.tar',
        'out_dim': 8,
        'x_axis': [-1,-1,1,1],
        'y_axis': [-1,1,-1,1],
        'tnf_type': 'tps',
        'feature_extraction': 'resnet101',
        'static_params': torch.tensor([[-1.0,-1.0,1.0,1.0,-1.0,1.0,-1.0,1.0]])
    },
    'tps_8p_no_occ':{
        'model_path': 'trained_models/tps_8p_no_occ/best_tps_8p_no_occ_tps_grid_lossresnet101.pth.tar',
        'out_dim': 16,
        'x_axis': [-1,-1,-1,0,0,1,1,1],
        'y_axis': [ -1,0,1,-1,1,-1,0,1],
        'tnf_type': 'tps',
        'feature_extraction': 'resnet101',
        'static_params': torch.tensor([[-1.0,-1.0,-1.0,0.0,0.0,1.0,1.0,1.0,1.0,0.0,-1.0]])
    },
    'tps_9p_no_occ':{
        'model_path': 'trained_models/tps_9p_no_occ/best_tps_9p_no_occ_tps_grid_lossresnet101.pth.tar',
        'out_dim': 18,
        'x_axis': [-1,-1,-1,0,0,0,1,1,1],
        'y_axis': [-1,0,1,-1,0,1,-1,0,1],
        'tnf_type': 'tps',
        'feature_extraction': 'resnet101',
        'static_params': torch.tensor([[-1.0,-1.0,-1.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0,0.0,-1.0]])
    },
    'affine_no_occ':{
        'model_path': 'trained_models/affine_no_occ/best_affine_no_occ_affine_grid_lossresnet101.pth.tar',
        'out_dim': 6,
        'x_axis': None,
        'y_axis': None,
        'tnf_type': 'aff',
        'feature_extraction': 'resnet101',
        'static_params':torch.tensor([[1.0,0.0,0.0,0.0,1.0,0.0]])
    },
    'hom_no_occ':{
        'model_path': 'trained_models/hom_no_occ/best_hom_no_occ_hom_grid_lossresnet101.pth.tar',
        'out_dim': 8,
        'x_axis': None,
        'y_axis': None,
        'tnf_type': 'hom',
        'feature_extraction': 'resnet101',
        'static_params':torch.tensor([[-1.0,-1.0,1.0,1.0,-1.0,1.0,-1.0,1.0]])
    },
    'hom_no_occ_custom':{
        'model_path': 'trained_models/hom_no_occ_custom/best_checkpoint_adam_hom_grid_lossresnet101.pth.tar',
        'out_dim': 8,
        'x_axis': None,
        'y_axis': None,
        'tnf_type': 'hom',
        'feature_extraction': 'resnet101',
        'static_params':torch.tensor([[-1.0,-1.0,1.0,1.0,-1.0,1.0,-1.0,1.0]])
    },
    'hom_interior_custom':{
        'model_path': 'trained_models/hom_interior/best_hom-interior_hom_grid_lossvgg.pth.tar',
        'out_dim': 8,
        'x_axis': None,
        'y_axis': None,
        'tnf_type': 'hom',
        'feature_extraction': 'vgg',
        'static_params':torch.tensor([[-1.0,-1.0,1.0,1.0,-1.0,1.0,-1.0,1.0]])
    },
    'hom_interior_resnet_custom':{
        'model_path': 'trained_models/hom-interior-resnet/best_hom-interior-resnet_hom_grid_lossvgg.pth.tar',
        'out_dim': 8,
        'x_axis': None,
        'y_axis': None,
        'tnf_type': 'hom',
        'feature_extraction': 'resnet101',
        'static_params':torch.tensor([[-1.0,-1.0,1.0,1.0,-1.0,1.0,-1.0,1.0]])
    },
    # Occlusion models
    'tps_4p_occ':{
        'model_path': 'trained_models/tps_4p_occ/best_tps_4p_occ_tps_grid_lossresnet101.pth.tar',
        'out_dim': 8,
        'x_axis': [-1,-1,1,1],
        'y_axis': [-1,1,-1,1],
        'tnf_type': 'tps',
        'feature_extraction': 'resnet101',
        'static_params': torch.tensor([[-1.0,-1.0,1.0,1.0,-1.0,1.0,-1.0,1.0]])
    },
    'tps_8p_occ':{
        'model_path': 'trained_models/tps_8p_occ/best_tps_8p_occ_tps_grid_lossresnet101.pth.tar',
        'out_dim': 16,
        'x_axis': [-1,-1,-1,0,0,1,1,1],
        'y_axis': [ -1,0,1,-1,1,-1,0,1],
        'tnf_type': 'tps',
        'feature_extraction': 'resnet101',
        'static_params': torch.tensor([[-1.0,-1.0,-1.0,0.0,0.0,1.0,1.0,1.0,1.0,0.0,-1.0]])
    },
    'tps_9p_occ':{
        'model_path': 'trained_models/tps_9p_occ/best_tps_9p_occ_tps_grid_lossresnet101.pth.tar',
        'out_dim': 18,
        'x_axis': [-1,-1,-1,0,0,0,1,1,1],
        'y_axis': [-1,0,1,-1,0,1,-1,0,1],
        'tnf_type': 'tps',
        'feature_extraction': 'resnet101',
        'static_params': torch.tensor([[-1.0,-1.0,-1.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0,0.0,-1.0]])
    },
    'affine_occ':{
        'model_path': 'trained_models/affine_occ/best_affine_occ_affine_grid_lossresnet101.pth.tar',
        'out_dim': 6,
        'x_axis': None,
        'y_axis': None,
        'tnf_type': 'aff',
        'feature_extraction': 'resnet101',
        'static_params':torch.tensor([[1.0,0.0,0.0,0.0,1.0,0.0]])
    },
    'hom_occ':{
        'model_path': 'trained_models/hom_correct_occ_0.5/best_checkpoint_adam_hom_grid_lossvgg.pth.tar',
        'out_dim': 8,
        'x_axis': None,
        'y_axis': None,
        'tnf_type': 'hom',
        # 'feature_extraction': 'resnet101',
        'feature_extraction': 'vgg',
        'static_params':torch.tensor([[-1.0,-1.0,1.0,1.0,-1.0,1.0,-1.0,1.0]])
    },
    # No occlusion -> occlusion models
    'tps_4p_no_occ_occ':{
        'model_path': 'trained_models/tps_4p_no_occ_occ/best_tps_4p_no_occ_occ_tps_grid_lossresnet101.pth.tar',
        'out_dim': 8,
        'x_axis': [-1,-1,1,1],
        'y_axis': [-1,1,-1,1],
        'tnf_type': 'tps',
        'feature_extraction': 'resnet101',
        'static_params': torch.tensor([[-1.0,-1.0,1.0,1.0,-1.0,1.0,-1.0,1.0]])
    },
    'tps_8p_no_occ_occ':{
        'model_path': 'trained_models/tps_8p_no_occ_occ/best_tps_8p_no_occ_occ_tps_grid_lossresnet101.pth.tar',
        'out_dim': 16,
        'x_axis': [-1,-1,-1,0,0,1,1,1],
        'y_axis': [ -1,0,1,-1,1,-1,0,1],
        'tnf_type': 'tps',
        'feature_extraction': 'resnet101',
        'static_params': torch.tensor([[-1.0,-1.0,-1.0,0.0,0.0,1.0,1.0,1.0,1.0,0.0,-1.0]])
    },
    'tps_9p_no_occ_occ':{
        'model_path': 'trained_models/tps_9p_no_occ_occ/best_tps_9p_no_occ_occ_tps_grid_lossresnet101.pth.tar',
        'out_dim': 18,
        'x_axis': [-1,-1,-1,0,0,0,1,1,1],
        'y_axis': [-1,0,1,-1,0,1,-1,0,1],
        'tnf_type': 'tps',
        'feature_extraction': 'resnet101',
        'static_params': torch.tensor([[-1.0,-1.0,-1.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0,0.0,-1.0]])
    },
    'affine_no_occ_occ':{
        'model_path': 'trained_models/affine_no_occ_occ/best_affine_no_occ_occ_affine_grid_lossresnet101.pth.tar',
        'out_dim': 6,
        'x_axis': None,
        'y_axis': None,
        'tnf_type': 'aff',
        'feature_extraction': 'resnet101',
        'static_params':torch.tensor([[1.0,0.0,0.0,0.0,1.0,0.0]])
    },
    'hom_no_occ_occ':{
        'model_path': 'trained_models/hom_no_occ_occ/best_hom_no_occ_occ_hom_grid_lossresnet101.pth.tar',
        'out_dim': 8,
        'x_axis': None,
        'y_axis': None,
        'tnf_type': 'hom',
        'feature_extraction': 'resnet101',
        'static_params':torch.tensor([[-1.0,-1.0,1.0,1.0,1.0,-1.0,1.0,-1.0]])
    },
    # Occlusion -> no occlusion models
    'tps_4p_occ_no_occ':{
        'model_path': 'trained_models/tps_4p_occ_no_occ/best_tps_4p_occ_no_occ_tps_grid_lossresnet101.pth.tar',
        'out_dim': 8,
        'x_axis': [-1,-1,1,1],
        'y_axis': [-1,1,-1,1],
        'tnf_type': 'tps',
        'feature_extraction': 'resnet101',
        'static_params':torch.tensor([ [-1.0,-1.0,1.0,1.0,-1.0,1.0,-1.0,1.0]])
    },
    'tps_8p_occ_no_occ':{
        'model_path': 'trained_models/tps_8p_occ_no_occ/best_tps_8p_occ_no_occ_tps_grid_lossresnet101.pth.tar',
        'out_dim': 16,
        'x_axis': [-1,-1,-1,0,0,1,1,1],
        'y_axis': [ -1,0,1,-1,1,-1,0,1],
        'tnf_type': 'tps',
        'feature_extraction': 'resnet101',
        'static_params':torch.tensor([ [-1.0,-1.0,-1.0,0.0,0.0,1.0,1.0,1.0,1.0,0.0,-1.0]])
    },
    'tps_9p_occ_no_occ':{
        'model_path': 'trained_models/tps_9p_occ_no_occ/best_tps_9p_occ_no_occ_tps_grid_lossresnet101.pth.tar',
        'out_dim': 18,
        'x_axis': [-1,-1,-1,0,0,0,1,1,1],
        'y_axis': [-1,0,1,-1,0,1,-1,0,1],
        'tnf_type': 'tps',
        'feature_extraction': 'resnet101',
        'static_params':torch.tensor([ [-1.0,-1.0,-1.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0,0.0,-1.0]])
    },
    'affine_occ_no_occ':{
        'model_path': 'trained_models/affine_occ_no_occ/best_affine_occ_no_occ_affine_grid_lossresnet101.pth.tar',
        'out_dim': 6,
        'x_axis': None,
        'y_axis': None,
        'tnf_type': 'aff',
        'feature_extraction': 'resnet101',
        'static_params':torch.tensor([[1.0,0.0,0.0,0.0,1.0,0.0]])
    },
    'hom_occ_no_occ':{
        'model_path': 'trained_models/hom_occ_no_occ/best_hom_occ_no_occ_hom_grid_lossresnet101.pth.tar',
        'out_dim': 8,
        'x_axis': None,
        'y_axis': None,
        'tnf_type': 'hom',
        'feature_extraction': 'resnet101',
        'static_params':torch.tensor([[-1.0,-1.0,1.0,1.0,-1.0,1.0,-1.0,1.0]])
    },
}

model_paths = { 'hom': 'trained_models/best_streetview_checkpoint_adam_hom_grid_loss_PAMI_usefull.pth.tar' ,
                'aff': 'trained_models/checkpoint_adam/best_checkpoint_adam_affine_grid_lossresnet101_occluded.pth.tar',
                'tps': 'trained_models/checkpoint_adam/best_checkpoint_adam_tps_grid_lossresnet101_occluded.pth.tar' }

model_out_dim = { 'hom': 8 ,
                'aff': 6,
                'tps': 18 }

MODEL_PATH = 'trained_models/best_streetview_checkpoint_adam_hom_grid_loss_PAMI_usefull.pth.tar'
# MODEL_PATH = 'trained_models/checkpoint_adam/checkpoint_adam_hom_grid_lossvgg.pth.tar'
model_hom = None

resize_cnn = GeometricTnf(out_h=240, out_w=240, use_cuda = False) 
normalize_tnf= Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

stitch_namespace = dict()


def extendAxis(axis):

    ext_size = len(axis)
    start = axis[0]
    end = axis[-1]

    dist = end-start
    step = dist / (len(axis)-1)

    # left_ext = np.arange(start=start,stop= (start-int(dist/2)),step=-step )
    # left_ext = left_ext[1:]
    
    left_ext = np.array( [ axis[0]-step*i for i in range(1,1+ext_size//2) ] )

    left_ext = np.flip(left_ext)

    right_ext = np.array( [ axis[-1]+step*i for i in range(1,1+ext_size//2) ] )

    # right_ext = np.arange(start=end,stop= (end+int(dist/2)),step=step )
    # right_ext = right_ext[1:]
    # right_ext = np.append(right_ext,right_ext[-1]+step)

    return np.concatenate([left_ext,axis,right_ext])

def preprocess_image(image):
    # convert to torch Variable
    image = np.expand_dims(image.transpose((2,0,1)),0)
    image = torch.Tensor(image.astype(np.float32)/255.0)
    image_var = Variable(image,requires_grad=False)

    # Resize image using bilinear sampling with identity affine tnf
    image_var = resize_cnn(image_var)
    
    # Normalize image
    image_var = normalize_image(image_var)
    
    return image_var

def loadModels():

    print("loading models")

    feature_extraction_cnn = 'vgg'

    model_hom = CNNGeometric(use_cuda=USE_CUDA,output_dim=8,feature_extraction_cnn=feature_extraction_cnn)

    checkpoint = torch.load(MODEL_PATH, map_location=lambda storage, loc: storage)
    checkpoint['state_dict'] = OrderedDict([(k.replace('vgg', 'model'), v) for k, v in checkpoint['state_dict'].items()])
    model_hom.load_state_dict(checkpoint['state_dict'])

    print("finished models load")

    return True,model_hom

def loadModel(feature_extraction: str = 'vgg', cnn_out_dim: int = 6, model_path: str = ''):

    model = CNNGeometric(use_cuda=USE_CUDA,output_dim=cnn_out_dim,feature_extraction_cnn=feature_extraction)

    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    checkpoint['state_dict'] = OrderedDict([(k.replace('vgg', 'model'), v) for k, v in checkpoint['state_dict'].items()])
    model.load_state_dict(checkpoint['state_dict'])

    print("finished models load")

    return model


def stitch( imgSrcP : str , imgTrgP: str ):

    success, model_hom = loadModels()

    if (success):

        source_image = io.imread(imgSrcP)
        target_image = io.imread(imgTrgP)

        source_image_var = preprocess_image(source_image)
        target_image_var = preprocess_image(target_image)

        if USE_CUDA:
            source_image_var = source_image_var.cuda()
            target_image_var = target_image_var.cuda()

        batch = {'source_image': source_image_var, 'target_image':target_image_var}
            
        model_hom.eval()

        theta_hom=model_hom(batch)
        print(theta_hom)

        h_matrix =  homography_mat_from_4_pts(theta_hom)
        src_h_matrix = h_matrix.detach().cpu().numpy().reshape((-1,3))

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

        ####Process source image
        boundBox = aling.maxBoundBox(src_h_matrix)

        h_points = aling.getImgPoints(boundBox,points)

        img = cv.imread(imgSrcP)
        img = cv.resize(img,(out_w,out_h))

        newImg=aling.applyHom(img,h_points,H_X_axis,np.flip(H_Y_axis),X_axis,np.flip(Y_axis),src_h_matrix)

        #Process image weights for blending

        weights = blend.genWieghts(out_w,out_h)

        h_weights=aling.applyHom(weights,h_points,H_X_axis,np.flip(H_Y_axis),X_axis,np.flip(Y_axis),src_h_matrix,singleDimension=True)

        #Add results to stack
        img_stack.append(newImg)
        w_stack.append(h_weights)


        ####Process target img
        trg_h_matrix = np.array( [ [1,0,-1],[0,1,0],[0,0,1] ] )

        boundBox = aling.maxBoundBox(trg_h_matrix)

        h_points = aling.getImgPoints(boundBox,points)

        img = cv.imread(imgTrgP)
        img = cv.resize(img,(out_h,out_w))

        newImg=aling.applyHom(img,h_points,H_X_axis,np.flip(H_Y_axis),X_axis,np.flip(Y_axis),trg_h_matrix,one2oneMaping=True)
        h_weights=aling.applyHom(weights,h_points,H_X_axis,np.flip(H_Y_axis),X_axis,np.flip(Y_axis),trg_h_matrix,one2oneMaping=True,singleDimension=True)

        img_stack.append(newImg)
        w_stack.append(h_weights)


        ### Blend results
        img_stack = np.array(img_stack)
        w_stack = np.array(w_stack)

        newImg = blend.blendImgs(img_stack,w_stack)

        cv.imwrite("result.jpg",newImg)
    
    else:

        print("Failed to load models")


def stitch(imgArray: List[str],dest) -> np.array:

    success, model_hom = loadModels()

    if (success):
        print("Models loaded")
        
        newImgSpace = blend.calcNewSpace((out_h,out_w,3),len(imgArray))
        # newWeightSpace = np.append(newImgSpace[:2],[1])

        imgStack = []
        weightStack = []

        baseImgPath = imgArray[0]
        base_h_matrix = np.array([ [1,0,0],[0,1,0],[0,0,1] ])        

        auxImg,auxWeights = processHom_str(baseImgPath,base_h_matrix,one2one=True)

        auxImg = blend.fitInSpace(auxImg,newImgSpace,(out_h,out_w),(0,1))
        auxWeights = blend.fitInSpace(auxWeights,newImgSpace,(out_h,out_w),(0,1))
        
        cv.imwrite("midle_img.jpg",auxImg)

        imgStack.append(auxImg)
        weightStack.append(auxWeights)

        baseImg = cv.imread(baseImgPath)

        for i in range(1,len(imgArray)):
            
            #Asuming 3 photos by strip
            y_coord = i % 3 

            x_coord = math.floor(i/3)

            srcImg = cv.imread(imgArray[i])

            h_matrix = getHomMatrix_np(srcImg,baseImg,model_hom)

            srcImg = cv.resize(srcImg,(out_h,out_w))
            
            auxImg,auxWeights = processHom_np(srcImg,h_matrix)
            
            if (y_coord == 0):
                y_start = math.floor(out_h/2)
                y_end = y_start+out_h

                x_start = math.floor(out_w/2)
                x_end = x_start+out_w

                baseImg = auxImg[y_start:y_end,x_start:x_end]

                y_coord = 1
            elif (y_coord ==1):
                y_coord = 0

            cv.imwrite("./cache/resultNoFit"+str(i)+".jpg",auxImg)
            
            auxImg = blend.fitInSpace(auxImg,newImgSpace,(out_h,out_w),(x_coord,y_coord))
            auxWeights = blend.fitInSpace(auxWeights,newImgSpace,(out_h,out_w),(x_coord,y_coord))


            cv.imwrite("./cache/resultFit"+str(i)+".jpg",auxImg)
            
            imgStack.append(auxImg)
            weightStack.append(auxWeights)

        imgStack = np.array(imgStack)
        weightStack = np.array(weightStack)

        blendedImg = blend.blendImgs(imgStack,weightStack,(out_h,out_w))

        if (not path.exists(dest)):
            makedirs(dest)

        y_dim = blendedImg.shape[0]
        x_dim = blendedImg.shape[0]*2

        resized = cv.resize(blendedImg,(x_dim,y_dim))

        filePath = path.join(dest,'result.png')

        cv.imwrite(filePath,resized)

        return filePath

    else:

        print("Failed to load models")
        sys.exit(-1)




#This method expands the image to the given shape wihout stretching it
# Zones are considered as:
#   [1]
#   [0]
#   [2]
# Where each zone its a third of the new space height
def expandImgHeihght(img: np.array, shape: np.array, position: int) -> np.array:

    if position == 0:
        position == 1
    
    if position == 1:
        position == 0

    start = math.ceil(shape[0]/3)*position
    end = start + img.shape[0]

    newImg = np.zeros(shape)

    newImg[start:end] = img

    return newImg


#Calculates the homography matrix using the trained model
# @dispatch(str,str,CNNGeometric,namespace= stitch_namespace)
def getHomMatrix_str(imgSrcPath: str, imgTrgPath: str,model: CNNGeometric)-> np.array:

    source_image = io.imread(imgSrcPath)
    target_image = io.imread(imgTrgPath)

    return getHomMatrix_np(source_image,target_image,model)

# @dispatch(np.ndarray,np.ndarray,CNNGeometric,namespace=stitch_namespace)
def getHomMatrix_np(imgSrc: np.ndarray,imgTrg: np.ndarray,model: CNNGeometric)-> np.array:

    # source_image = io.imread(imgSrcPath)
    # target_image = io.imread(imgTrgPath)

    source_image = imgSrc
    target_image = imgTrg

    source_image_var = preprocess_image(source_image)
    target_image_var = preprocess_image(target_image)

    if USE_CUDA:
        source_image_var = source_image_var.cuda()
        target_image_var = target_image_var.cuda()

    batch = {'source_image': source_image_var, 'target_image':target_image_var}
        
    model.eval()

    theta=model(batch)
    print(theta)

    h_matrix =  homography_mat_from_4_pts(theta)
    
    h_matrix = h_matrix.detach().cpu().numpy().reshape((-1,3))
    
    return h_matrix

#Calculates the homography matrix using the trained model
# @dispatch(str,str,CNNGeometric,namespace= stitch_namespace)
def getTheta_str(imgSrcPath: str, imgTrgPath: str,model: CNNGeometric)-> np.array:

    source_image = io.imread(imgSrcPath)
    target_image = io.imread(imgTrgPath)

    return getTheta_np(source_image,target_image,model)

# @dispatch(np.ndarray,np.ndarray,CNNGeometric,namespace=stitch_namespace)
def getTheta_np(imgSrc: np.ndarray,imgTrg: np.ndarray,model: CNNGeometric)-> np.array:

    # source_image = io.imread(imgSrcPath)
    # target_image = io.imread(imgTrgPath)

    source_image = imgSrc
    target_image = imgTrg

    source_image_var = preprocess_image(source_image)
    target_image_var = preprocess_image(target_image)

    if USE_CUDA:
        source_image_var = source_image_var.cuda()
        target_image_var = target_image_var.cuda()

    batch = {'source_image': source_image_var, 'target_image':target_image_var}
        
    model.eval()

    theta=model(batch)
    print(theta)
    
    return theta


# @dispatch(str,np.ndarray,bool,namespace=stitch_namespace)
def processHom_str(imgPath: str, h_matrix: np.ndarray,one2one: bool = False)-> np.array :

    img = cv.imread(imgPath)
    img = cv.resize(img,(out_w,out_h))

    return processHom_np(img,h_matrix,one2one)


#Process the homography to get the image on an expanded space to fit the whole image plus the weights to do the blending
def processHom_np(img: np.ndarray, h_matrix: np.ndarray,one2one: bool = False)-> np.array :

    # img = cv.imread(imgPath)
    # img = cv.resize(img,(out_w,out_h))

    #Generate axis in normalized space
    X_axis = np.linspace(startNormSpace,endNormSpace,out_w)
    Y_axis = np.linspace(startNormSpace,endNormSpace,out_h)

    #Extend axis to fit transformed img
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

    ####Process source image
    boundBox = aling.maxBoundBox(h_matrix)

    h_points = aling.getImgPoints(boundBox,points)

    newImg=aling.applyHom(img,h_points,H_X_axis,np.flip(H_Y_axis),X_axis,np.flip(Y_axis),h_matrix,one2oneMaping=one2one)

    #Process image weights for blending

    weights = blend.genWieghts(out_w,out_h)

    h_weights=aling.applyHom(weights,h_points,H_X_axis,np.flip(H_Y_axis),X_axis,np.flip(Y_axis),h_matrix,singleDimension=True,one2oneMaping=one2one)

    return newImg,h_weights

#Process the homography to get the image on an expanded space to fit the whole image plus the weights to do the blending
def processAff_np(img: np.ndarray, theta: np.ndarray,one2one: bool = False)-> np.array :

    # img = cv.imread(imgPath)
    # img = cv.resize(img,(out_w,out_h))

    #Generate axis in normalized space
    X_axis = np.linspace(startNormSpace,endNormSpace,out_w)
    Y_axis = np.linspace(startNormSpace,endNormSpace,out_h)

    #Extend axis to fit transformed img
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

    ####Process source image
    boundBox = aling.maxBoundBox(h_matrix)

    h_points = aling.getImgPoints(boundBox,points)

    newImg=aling.applyHom(img,h_points,H_X_axis,np.flip(H_Y_axis),X_axis,np.flip(Y_axis),h_matrix,one2oneMaping=one2one)

    #Process image weights for blending

    weights = blend.genWieghts(out_w,out_h)

    h_weights=aling.applyHom(weights,h_points,H_X_axis,np.flip(H_Y_axis),X_axis,np.flip(Y_axis),h_matrix,singleDimension=True,one2oneMaping=one2one)

    return newImg,h_weights



# process_map_str = { 'hom' : processHom_str,
#                     'aff' : processAff_str,
#                     'tps' : processTps_str}

# process_map_np = { 'hom' : processHom_np,
#                     'aff' : processAff_np,
#                     'tps' : processTps_np}



def fitForTransform(img):

    h = img.shape[0]
    w = img.shape[1]
    c = img.shape[2]

    new_img = np.zeros((h*2,w*2,c),dtype=int)

    new_img[ (h//2):(h//2)+h , (w//2):(w//2)+w  ] = img

    return new_img

def na_prep_method(img_src,img_tgt,y_coord,x_dir):
    return img_src,img_tgt

# The occlusion assumes 3 imgs by strip and 50% overlap between src and tgt 
def occlusion_prep_method(img_src,img_tgt, y_coord,x_dir):

    height = img_src.shape[0]
    width = img_src.shape[1]

    y_coord = (y_coord - 1)*-1

    y_shift = y_coord * (height//2)

    x_shift = 0

    if (y_coord == 0 ):
        x_shift = (-width//2)*x_dir

    M = np.float32( [[1,0,x_shift],[0,1,y_shift]])

    aux_tgt = cv.warpAffine(img_tgt,M,(width,height))

    cv.imwrite('aux_tgt.jpg',aux_tgt)

    mask = np.zeros( img_src.shape[:2],np.uint8)
    mask.fill(255)

    mask = cv.warpAffine(mask,M,(width,height))

    cv.imwrite('mask.jpg',mask)

    aux_src = cv.bitwise_and(img_src,img_src,mask=mask)

    return aux_src,aux_tgt

grid_gen_map = { 'hom' : StitchingHomographyGridGen,
                    'aff' : StitchingAffineGridGen,
                    'tps' : StitchingTpsGridGen}

prep_method_map = { 'NA': na_prep_method,
                    'occlusion': occlusion_prep_method }

def plotGrid(grid: np.ndarray,figName:str):
    fig, ax = plt.subplots()
    ax.matshow(grid, cmap='gray')

    # for i in range(grid.shape[1]):
    #     for j in range(grid.shape[0]):
    #         c = grid[j, i]
    #         ax.text(i, j, str(c), va='center', ha='center')

    plt.savefig("./plots/{}.jpg".format(figName))
    plt.close()
    pass


class ImageStitcher():

    def __init__(self,modelQueue: List[str],prep_method: str = 'NA'):
        
        # loadModels()
        
        model_params_queue = [ models_paramas[model] for model in modelQueue ] 
        self.model_name_queue = modelQueue
        self.tnf_type_queue = [ model_params['tnf_type'] for model_params in model_params_queue]
        self.x_axis_queue = [model_params['x_axis'] for model_params in  model_params_queue]
        self.y_axis_queue = [ model_params['y_axis'] for model_params in model_params_queue] 

        self.model_queue = [ loadModel(feature_extraction=model_params['feature_extraction'],cnn_out_dim=model_params['out_dim'],
                                model_path=model_params['model_path']) for model_params in model_params_queue]
        self.grid_gen_queue = [ grid_gen_map[tnf_type] for tnf_type in self.tnf_type_queue ] 
        self.prep_imgs = prep_method_map[prep_method]
    
    #NOTE: with the changes to the queue of models this method wont work
    #       stays here just to reference
    def stitch(self,imgArray: List[str], dest) -> np.ndarray :

        print("Starting stitch")

        if self.tnf_type == 'tps':
            grid_generator = self.grid_gen(img_w=out_w,img_h=out_h,use_regular_grid=False,x_axis_coords=self.x_axis,y_axis_coords=self.y_axis,
                                            use_cuda=USE_CUDA)
        else:
            grid_generator = self.grid_gen(img_w=out_w,img_h=out_h,
                                            use_cuda=USE_CUDA)

        newImgSpace = blend.calcNewSpace((out_h,out_w,3),len(imgArray))

        imgStack = []
        weightStack = []

        base_img = cv.imread(imgArray[0])

        base_img = cv.resize(base_img,(out_h,out_w))


        new_base_img = np.zeros((out_h*2,out_w*2,3),dtype=int)

        new_base_img[ (out_h//2):(out_h//2)+out_h , (out_w//2):(out_w//2)+out_w  ] = base_img

        cv.imwrite("./cache/resultNoFit0.jpg",new_base_img)
        
        # new_base_img = fitForBlending(base_img)

        new_base_weights = np.zeros((out_h*2,out_w*2,3),dtype=float)
        weights = blend.genWieghts(out_w,out_h)
        new_base_weights[ (out_h//2):(out_h//2)+out_h , (out_w//2):(out_w//2)+out_w  ] = weights
        # print(new_base_weights.max)
        # new_base_weights = fitForBlending(weights)

        auxImg = blend.fitInSpace(new_base_img,newImgSpace,(out_h,out_w),(0,1))
        cv.imwrite("./cache/resultFit0.jpg",auxImg)
        auxWeights = blend.fitInSpace(new_base_weights,newImgSpace,(out_h,out_w),(0,1))
        
        np.savetxt("weigths0.csv",new_base_weights[:,:,0],delimiter=",")
        

        cv.imwrite("midle_img.jpg",auxImg)

        imgStack.append(auxImg)
        weightStack.append(auxWeights)

        baseImgPath = imgArray[0]
        baseImg = cv.imread(baseImgPath)

        
        for i in range(1,len(imgArray)):
            
            #Asuming 3 photos by strip
            y_coord = i % 3 

            x_coord = math.floor(i/3)

            srcImg = cv.imread(imgArray[i])
            
            srcImg = cv.resize(srcImg,(out_h,out_w))

            prep_src,prep_img = self.prep_imgs(srcImg,baseImg,y_coord)


            # h_matrix = getHomMatrix_np(srcImg,baseImg,model_hom)
            theta = getTheta_np(prep_src,prep_img,self.model)        

            grid = grid_generator(theta)

            weights = blend.genWieghts(out_w, out_h)

            # srcImg = fitForTransform(srcImg)
            # weights = fitForTransform(weights)
            
            grid_np = grid[0].detach().cpu().numpy()

            auxImg = flex_grid_sample(cv.transpose(srcImg),grid_np,out_h,out_w)
            auxWeights = flex_grid_sample(cv.transpose(weights),grid_np,out_h,out_w)

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
            # auxWeights= auxWeights[0]
            # auxWeights = auxWeights.transpose(0,1).transpose(1,2).detach().cpu().numpy()

            # auxImg,auxWeights = self.process_map_np(srcImg,h_matrix)
            
            if (y_coord == 0):
                y_start = math.floor(out_h/2)
                y_end = y_start+out_h

                x_start = math.floor(out_w/2)
                x_end = x_start+out_w

                baseImg = auxImg[y_start:y_end,x_start:x_end]

                cv.imwrite("./cache/baseImg"+str(i)+".jpg",baseImg)

                y_coord = 1
            elif (y_coord ==1):
                y_coord = 0
                cv.imwrite("./cache/baseImg"+str(i)+".jpg",baseImg)

            cv.imwrite("./cache/resultNoFit"+str(i)+".jpg",auxImg)
            
            auxImg = blend.fitInSpace(auxImg,newImgSpace,(out_h,out_w),(x_coord,y_coord))
            auxWeights = blend.fitInSpace(auxWeights,newImgSpace,(out_h,out_w),(x_coord,y_coord))


            cv.imwrite("./cache/resultFit"+str(i)+".jpg",auxImg)
            
            imgStack.append(auxImg)
            weightStack.append(auxWeights)

        imgStack = np.array(imgStack)
        weightStack = np.array(weightStack)

        blendedImg = blend.blendImgs(imgStack,weightStack,(out_h,out_w))

        if (not path.exists(dest)):
            makedirs(dest)

        y_dim = blendedImg.shape[0]
        x_dim = blendedImg.shape[0]*2

        resized = cv.resize(blendedImg,(x_dim,y_dim))

        filePath_1 = path.join(dest,'result'+self.model_name+'.jpg')
        filePath = path.join(dest,'result.png')
        
        cv.imwrite(filePath,resized)    
        cv.imwrite(filePath_1,resized)    

        return filePath
    # def getTheta(self,srcImg,tgtImg):
    

    def stitchv2(self,pathArray: List[str], dest:str, returnImgsList: bool = False,x_axis_only:bool = False) -> np.ndarray :

        print("Starting stitch")

        grid_generator_queue = []

        for i, tnf_type in enumerate(self.tnf_type_queue):
            if tnf_type == 'tps':
                grid_generator = self.grid_gen_queue[i](img_w=out_w,img_h=out_h,use_regular_grid=False,x_axis_coords=self.x_axis_queue[i]
                                                ,y_axis_coords=self.y_axis_queue[i],use_cuda=USE_CUDA)
            else:
                grid_generator = self.grid_gen_queue[i](img_w=out_w,img_h=out_h,
                                                use_cuda=USE_CUDA)
            grid_generator_queue.append(grid_generator) 

        
        imgArray = [ Image(path,i,transform_params_stack=[],resize=True) for i,path in enumerate(pathArray) ]
        
        imgPairs = self.buildPairs(imgArray,x_axis_only)

        for model_name in reversed(self.model_name_queue):
            model_params = models_paramas[model_name]
            imgPairs[0].imgTgt.transform_stack.append(model_params['static_params'])

        imgRegistered = set([imgPairs[0].imgTgt])

        while len(imgRegistered) < len(imgArray):

            for imgPair in imgPairs:
                
                if (imgPair.imgTgt in imgRegistered) and (imgPair.imgSrc not in imgRegistered):
                
                    
                    src_img = imgPair.imgSrc
                    tgt_img = imgPair.imgTgt
                    src_img_array = src_img.img
                    tgt_img_array = tgt_img.img
                    y_coord = imgPair.y_src_coord
                    x_coord = imgPair.x_src_coord
                    x_dir = imgPair.x_src_dir

                    for theta in tgt_img.transform_stack:
                        src_img.transform_stack.append(theta)

                    theta_queue = []
                    for i,currentModel in enumerate(self.model_queue):
                        prep_src,prep_img = self.prep_imgs(src_img_array,tgt_img_array,y_coord,x_dir)
                        theta = getTheta_np(prep_src,prep_img,currentModel)        

                        #Remember that the generated grid is double size of img
                        grid = grid_generator_queue[i](theta)

                        grid_np = grid[0].detach().cpu().numpy()


                        # This function retursn an image double the size to make sure there is no information loss
                        auxImg = flex_grid_sample(cv.transpose(src_img_array),grid_np,out_h,out_w)

                        src_img_array = auxImg[out_h//2 : (out_h//2)+out_h , out_w//2:(out_w//2)+out_w ] 

                        # cv.imwrite("./cache/baseImg-{}-{}.jpg".format(src_img.index,str(i)),src_img_array)

                        # grid_i = grid_generator_queue[i].inverse(theta)

                        theta_queue.append(theta)

                    theta_queue.reverse()
                    src_img.transform_stack = src_img.transform_stack + theta_queue
                    imgRegistered.add(src_img)

        if returnImgsList:
            return imgArray

        pano = self.transformAndBlend(imgArray,x_axis_only)

        
        y_dim = pano.shape[0]
        x_dim = pano.shape[0]*2

        resized = cv.resize(pano,(x_dim,y_dim))

        return resized
        
        # # filePath_1 = path.join(dest,'result'+self.model_name+'.jpg')
        # filePath = path.join(dest,'{}result.png'.format(self.model_name_queue[0]))
        
        # cv.imwrite(filePath,resized)    
        # # cv.imwrite(filePath_1,resized)    

        # return filePath

        # return pano    


    
    def transformAndBlend(self, imgArray,x_axis_only:bool= False):
        
        grid_generator_queue = []

        for i, tnf_type in enumerate(self.tnf_type_queue):
            if tnf_type == 'tps':
                grid_generator = self.grid_gen_queue[i](img_w=out_w,img_h=out_h,use_regular_grid=False,x_axis_coords=self.x_axis_queue[i]
                                                ,y_axis_coords=self.y_axis_queue[i],use_cuda=USE_CUDA)
            else:
                grid_generator = self.grid_gen_queue[i](img_w=out_w,img_h=out_h,
                                                use_cuda=USE_CUDA)
            grid_generator_queue.append(grid_generator) 

        newImgSpace = blend.calcNewSpace((out_h,out_w,3),len(imgArray),x_axis_only=x_axis_only)

        xCoordStack = []
        yCoordStack = []
        weightStack = []
        
        for j,img in enumerate(imgArray):
            
            img.transform_stack.reverse()

            # if (len(img.transform_stack) == 2 ):
            #     img.transform_stack = [ torch.tensor([[-0.9927,-0.90315, 0.9724, 1.0246, -0.79435 , 0.9116, -0.8986, 0.89265]]),img.transform_stack[-1]]

            weights = blend.genWieghts(out_w, out_h)

            for i,theta in enumerate(img.transform_stack):

                transformation_id = i%len(self.model_queue)

                if (USE_CUDA):
                    theta = theta.cuda()

                grid = grid_generator_queue[transformation_id](theta)

                grid_np = grid[0].detach().cpu().numpy()

                # plotGrid(weights[:,:,0],"before_{}_{}".format(j,i))

                weights = flex_grid_sample(cv.transpose(weights),grid_np,out_h,out_w)

                # plotGrid(weights[:,:,0],"afterE_{}_{}".format(j,i))

            
            y_coord = img.index % 3
            x_coord = math.floor(img.index / 3)

            if (x_axis_only):
                y_coord=1
                x_coord=img.index

            if (y_coord == 0):
                y_start = math.floor(out_h/2)
                y_end = y_start+out_h

                x_start = math.floor(out_w/2)
                x_end = x_start+out_w
                # cv.imwrite("./cache/baseImg"+str(i)+".jpg",baseImg)

                y_coord = 1
            elif (y_coord ==1):
                y_coord = 0
                # cv.imwrite("./cache/baseImg"+str(i)+".jpg",baseImg)

            # cv.imwrite("./cache/resultNoFit"+str(img.index)+".jpg",auxImg)
            
            # auxImg = blend.fitInSpace(auxImg,newImgSpace,(out_h,out_w),(x_coord,y_coord))
            auxWeights = blend.fitInSpace(weights,newImgSpace,(out_h,out_w),(x_coord,y_coord))


            # cv.imwrite("./cache/resultFit"+str(img.index)+".jpg",auxImg)
            
            # imgStack.append(auxImg)
            weightStack.append(auxWeights)
            xCoordStack.append(x_coord)
            yCoordStack.append(y_coord)


        # for i,matrix in enumerate(weightStack):
        #     plotGrid(matrix[:,:,0],"fullW_{}".format(i))

        test = np.array(weightStack)

        test = test[:,:,:,0]

        weights_matrix = np.transpose(test, (1, 2, 0))

        weights_maxes = np.max(weights_matrix, axis=2)[:, :, np.newaxis]
        max_weights_matrix = np.where(
            np.logical_and(weights_matrix == weights_maxes, weights_matrix > 0), 1.0, 0.0
        )

        max_weights_matrix = np.transpose(max_weights_matrix, (2, 0, 1))
# 

        #Adding axis to better compatibility with operations

        max_weights_matrix = np.expand_dims(max_weights_matrix,3)

        # for i,matrix in enumerate(max_weights_matrix):
        #     plotGrid(matrix[:,:,0],"max_{}".format(i))

        croped_weights = []

        for i,weights_mask in enumerate(max_weights_matrix):
            
            x_coord = xCoordStack[i]
            y_coord = yCoordStack[i]

            
            croped_w =blend.cropFromSpace(weights_mask,(out_w*2,out_h*2),(out_w,out_h),(x_coord,y_coord)) 

            croped_weights.append(croped_w)

        # for i,matrix in enumerate(croped_weights):
        #     plotGrid(matrix[:,:,0],"croped_max_{}".format(i))

        # max_weights_matrix = max_weights_matrix[:,out_h//2:(out_h//2)+out_h,out_w//2:(out_w//2)+out_w]

        for i,img in enumerate(imgArray):
            
            weights = croped_weights[i]

            if weights.shape[2] == 1:
                #Repeat las axis so transpose dont mess up things 
                weights = np.repeat(weights,3,2)
            
            img.transform_stack.reverse()

            for j,theta in enumerate(img.transform_stack):

                transformation_id = j%len(self.model_queue)

                if (USE_CUDA):
                    theta = theta.cuda()

                grid = grid_generator_queue[transformation_id].inverse(theta)

                grid_np = grid[0].detach().cpu().numpy()

                # plotGrid(weights[:,:,0],"before_{}_{}".format(j,i))

                weights = flex_grid_sample(cv.transpose(weights),grid_np,out_h,out_w)

            weights = weights[(out_h//2):(out_h//2)+out_h,(out_w//2):(out_w//2)+out_w]            
            croped_weights[i]= weights

        # for i,matrix in enumerate(croped_weights):
        #     plotGrid(matrix[:,:,0],"croped_max_i_{}".format(i))

        # test = np.expand_dims(test,axis = 2)        grid_generator_queue = []

        for i, tnf_type in enumerate(self.tnf_type_queue):
            if tnf_type == 'tps':
                grid_generator = self.grid_gen_queue[i](img_w=out_w,img_h=out_h,use_regular_grid=False,x_axis_coords=self.x_axis_queue[i]
                                                ,y_axis_coords=self.y_axis_queue[i],use_cuda=USE_CUDA)
            else:
                grid_generator = self.grid_gen_queue[i](img_w=out_w,img_h=out_h,
                                                use_cuda=USE_CUDA)
            grid_generator_queue.append(grid_generator) 

        print("building bands")

        blurred_weights = [[cv.GaussianBlur(croped_weights[i], (0, 0), 2 * sigma) for i in range(len(imgArray))]]
        sigma_images = [cv.GaussianBlur(image.img, (0, 0), sigma) for image in imgArray]
        bands = [
            [
                np.where(
                    imgArray[i].img.astype(np.int64) - sigma_images[i].astype(np.int64) > 0,
                    imgArray[i].img - sigma_images[i],
                    0,
                )
                for i in range(len(imgArray))
            ]
        ]

        for k in range(1, num_bands - 1):
            sigma_k = np.sqrt(2 * k + 1) * sigma
            blurred_weights.append(
                [cv.GaussianBlur(blurred_weights[-1][i], (0, 0), sigma_k) for i in range(len(imgArray))]
            )

            old_sigma_images = sigma_images

            sigma_images = [
                cv.GaussianBlur(old_sigma_image, (0, 0), sigma_k)
                for old_sigma_image in old_sigma_images
            ]
            bands.append(
                [
                    np.where(
                        old_sigma_images[i].astype(np.int64) - sigma_images[i].astype(np.int64) > 0,
                        old_sigma_images[i] - sigma_images[i],
                        0,
                    )
                    for i in range(len(imgArray))
                ]
            )

        blurred_weights.append([cv.GaussianBlur(blurred_weights[-1][i], (0, 0), sigma_k) for i in range(len(imgArray))])
        bands.append([sigma_images[i] for i in range(len(imgArray))])

        panorama = np.zeros(newImgSpace)

        print("building panoramas")
        for k in range(0, num_bands):
            panorama += self.build_band_panorama(imgArray, blurred_weights[k], bands[k], newImgSpace,grid_generator_queue,x_axis_only=x_axis_only)
            panorama[panorama < 0] = 0
            panorama[panorama > 255] = 255
            cv.imwrite("./plots/band{}.jpg".format(k),panorama)

        return panorama

        print("checkpoint")
    # def processImage(imgSrc,imgTgt,model )        

    

    def buildPairs(self,imgArray: List[Image], x_axis_only:bool = False)->List[PairImage]:
        
        #Calculate index of the most center image in the list
        base_index  = ((math.ceil(len(imgArray) / 3)//2))*3

        if x_axis_only:
            base_index = math.ceil(len(imgArray))//2

        imgPairs = []

        current_center = base_index

        
        if (not x_axis_only):

            for i in range(base_index+1,len(imgArray)):
                
                y_src_coord = i%3
                x_src_coord = math.floor(i/3)

                imgPairs.append( PairImage(
                                    imgSrc=imgArray[i],
                                    imgTgt=imgArray[current_center],
                                    y_src_coord=y_src_coord,
                                    x_src_coord=x_src_coord,
                                    x_src_dir= 1 ) )

                
                if ( i - current_center == 3): 
                    current_center = i
        else:

            for i in range(base_index+1,len(imgArray)):
                
                y_src_coord = 1
                x_src_coord = i

                imgPairs.append( PairImage(
                                    imgSrc=imgArray[i],
                                    imgTgt=imgArray[i-1],
                                    y_src_coord=y_src_coord,
                                    x_src_coord=x_src_coord,
                                    x_src_dir= 1 ) )


        current_center = base_index

        if (not x_axis_only):

            for i in range(base_index-3,-1,-3):

                y_src_coord = i%3
                x_src_coord = math.floor(i/3)
                
                
                imgPairs.append( PairImage(
                                    imgSrc=imgArray[i],
                                    imgTgt=imgArray[i+3],
                                    y_src_coord=y_src_coord,
                                    x_src_coord=x_src_coord,
                                    x_src_dir= -1 ) )

                for k in range(i+1,i+3):

                    y_src_coord = k%3
                    x_src_coord = math.floor(k/3)
                    
                    imgPairs.append( PairImage(
                                        imgSrc=imgArray[k],
                                        imgTgt=imgArray[i],
                                        y_src_coord=y_src_coord,
                                        x_src_coord=x_src_coord,
                                        x_src_dir= -1 ) )
        else:
            
            for i in range(base_index-1,-1,-1):

                y_src_coord = 1
                # x_src_coord = math.floor(i/3)
                x_src_coord = i

                imgPairs.append( PairImage(
                                    imgSrc=imgArray[i],
                                    imgTgt=imgArray[i+1],
                                    y_src_coord=y_src_coord,
                                    x_src_coord=x_src_coord,
                                    x_src_dir= -1 ) )

                

            # if ( current_center - i== 3): 
            #     current_center = i

        return imgPairs

    def build_band_panorama(
        self,
        images: List[Image],
        weights: List[np.ndarray],
        bands: List[np.ndarray],
        size: Tuple[int, int],
        grid_generator_queue,
        x_axis_only:str = False
    ) -> np.ndarray:
        """
        Build a panorama from the given bands and weights matrices.
        The images are needed for their homographies.

        Parameters
        ----------
        images : List[Image]
            Images to build the panorama from.
        weights : List[np.ndarray]
            Weights matrices for each image.
        bands : List[np.ndarray]
            Bands for each image.
        offset : np.ndarray
            Offset matrix.
        size : Tuple[int, int]
            Size of the panorama.

        Returns
        -------
        panorama : np.ndarray
            Panorama for the given bands and weights.
        """
        pano_weights = np.zeros(size)
        pano_bands = np.zeros(size)

        for i, image in enumerate(images):
            
            image.transform_stack.reverse()

            img_weights = weights[i]
            img_band = bands[i]

            for j,theta in enumerate(image.transform_stack):

                transformation_id = j%len(self.model_queue)

                if (USE_CUDA):
                    theta = theta.cuda()

                grid = grid_generator_queue[transformation_id](theta)

                grid_np = grid[0].detach().cpu().numpy()

                # plotGrid(weights[:,:,0],"before_{}_{}".format(j,i))

                img_weights= flex_grid_sample(cv.transpose(img_weights),grid_np,out_h,out_w)
                img_band= flex_grid_sample(cv.transpose(img_band),grid_np,out_h,out_w)

            
            y_coord = image.index % 3
            x_coord = math.floor(image.index / 3)

            if x_axis_only:
                y_coord = 1
                x_coord = image.index

            if (y_coord == 0):
                y_start = math.floor(out_h/2)
                y_end = y_start+out_h

                x_start = math.floor(out_w/2)
                x_end = x_start+out_w
                # cv.imwrite("./cache/baseImg"+str(i)+".jpg",baseImg)

                y_coord = 1
            elif (y_coord ==1):
                y_coord = 0
                # cv.imwrite("./cache/baseImg"+str(i)+".jpg",baseImg)

            # cv.imwrite("./cache/resultNoFit"+str(img.index)+".jpg",auxImg)
            
            # auxImg = blend.fitInSpace(auxImg,newImgSpace,(out_h,out_w),(x_coord,y_coord))
            weights_at_scale = blend.fitInSpace(img_weights,size,(out_h,out_w),(x_coord,y_coord))
            band_at_scale = blend.fitInSpace(img_band,size,(out_h,out_w),(x_coord,y_coord))

            pano_weights += weights_at_scale
            pano_bands += weights_at_scale*band_at_scale 


        return np.divide(
            pano_bands, pano_weights, where=pano_weights != 0
        ) 