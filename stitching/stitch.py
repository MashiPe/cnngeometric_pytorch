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
from typing import Any, List
from multipledispatch import dispatch
from numpy.typing import ArrayLike


out_w = 240
out_h = 240
startNormSpace= -1
endNormSpace = 1

USE_CUDA = torch.cuda.is_available()

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

def loadModel(feature_extraction: str = 'vgg', geotnf: str = 'hom'):

    model = CNNGeometric(use_cuda=USE_CUDA,output_dim=model_out_dim[geotnf],feature_extraction_cnn=feature_extraction)

    checkpoint = torch.load(model_paths[geotnf], map_location=lambda storage, loc: storage)
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

def na_prep_method(img_src,img_tgt,y_coord):
    return img_src,img_tgt

# The occlusion assumes 3 imgs by strip and 50% overlap between src and tgt 
def occlusion_prep_method(img_src,img_tgt, y_coord):

    height = img_src.shape[0]
    width = img_src.shape[1]

    y_coord = (y_coord - 1)*-1

    y_shift = y_coord * (height//2)

    x_shift = 0

    if (y_coord == 0 ):
        x_shift = -width//2

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

class ImageStitcher():

    def __init__(self,geotnf: str,feature_extraction: str,prep_method: str = 'NA'):
        
        # loadModels()
        self.model = loadModel(feature_extraction=feature_extraction,geotnf=geotnf)
        self.grid_gen = grid_gen_map[geotnf]
        self.prep_imgs = prep_method_map[prep_method]
    
    def stitch(self,imgArray: List[str], dest) -> np.ndarray :

        print("Starting stitch")
        
        grid_generator = self.grid_gen(img_w=out_w,img_h=out_h,out_h=out_h*2,out_w=out_w*2)

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

        filePath = path.join(dest,'result.png')
        
        cv.imwrite(filePath,resized)    

        return filePath
    # def getTheta(self,srcImg,tgtImg):
        
