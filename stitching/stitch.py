from argparse import Namespace
from os import makedirs, path
import util.align_util as aling
import util.blend_util as blend
import numpy as np
import math
from util.torch_util import expand_dim
import torch
from torch.autograd import Variable
import cv2 as cv
from model.cnn_geometric_model import CNNGeometric
from collections import OrderedDict
from geotnf.transformation import GeometricTnf
from torchvision.transforms import Normalize
from image.normalization import normalize_image
from skimage import io
import sys
from geotnf.transformation import homography_mat_from_4_pts
from typing import Any, List
from multipledispatch import dispatch
from numpy.typing import ArrayLike


out_w = 240
out_h = 240
startNormSpace= -1
endNormSpace = 1

USE_CUDA = torch.cuda.is_available()
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

# @dispatch(str,np.ndarray,bool,namespace=stitch_namespace)
def processHom_str(imgPath: str, h_matrix: np.ndarray,one2one: bool = False)-> np.array :

    img = cv.imread(imgPath)
    img = cv.resize(img,(out_w,out_h))

    return processHom_np(img,h_matrix,one2one)


#Process the homography to get the image on an expanded space to fit the whole image plus the weights to do the blending
# @dispatch(np.ndarray,np.ndarray,bool,namespace=stitch_namespace)
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