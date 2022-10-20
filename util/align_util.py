import math
from os import cpu_count
from timeit import repeat
import numpy as np
from shapely.geometry import Polygon,Point
import scipy.ndimage as ndimage
from sklearn.preprocessing import normalize
from multiprocessing import Pool
from numba import cuda,jit
import cv2 as cv
from util.gpu_polygon import PolyV2

from util.polygon import Poly

def lineLineIntersection(A, B, C, D):
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

# Hmatrix is the homography as a numpy array
# Asuming Hmatrix as form:
#   h1 h2 h3      
#   h4 h5 h6
#   h7 h8 h9
# Calculate corresponding points as:
# k = h7*x_b+h8*y_b+h9
# x_0 = h1*x_b+h2*y_b+h3
# y_0 = h4*x_b+h5*y_b+h6

# x_a = x_0/k
# y_a = y_0/k

# Returns the bound box asuming a normalized space from -1 to 1
# Calculate the boundbox from the matrix:
#   [ [-1,1],
#     [1,1],
#     [1,-1],
#     [-1-1]]
def maxBoundBox(Hmatrix):

    # HmatrixI = np.linalg.inv(Hmatrix)
    coordX = np.array([[-1],[1],[1],[-1]],dtype=float)
    coordY = np.array([[1],[1],[-1],[-1]],dtype=float)

    k = Hmatrix[2][0]*coordX + Hmatrix[2][1]*coordY + Hmatrix[2][2]
    x_0 = Hmatrix[0][0]*coordX + Hmatrix[0][1]*coordY + Hmatrix[0][2]
    y_0 = Hmatrix[1][0]*coordX + Hmatrix[1][1]*coordY + Hmatrix[1][2]

    x_a = x_0/k
    y_a = y_0/k

    boundBox = np.concatenate((x_a,y_a),axis=1)

    return boundBox

def parrallelCheck(index,point,poly):

    return poly.contains(point)
    # return poly.contains(Point(point))

# Takes coordinate points from a numpy array and returns points that are contained within 
# the bounding box.
def getImgPoints(boundBox, points):

    # poly = Polygon(boundBox)


    center_point = lineLineIntersection(boundBox[0],boundBox[2],boundBox[3],boundBox[1])
    
    # poly = Poly(boundBox,center_point)

    # # mask = np.empty(points.shape[0])

    # with Pool(cpu_count()) as p:
    #     mask = p.starmap(parrallelCheck,zip(range(points.shape[0]),points,np.full((points.shape[0]),poly)))

    poly = PolyV2(boundBox,center_point)

    valid_points = poly.containsPoints(points)

    # valid_points = np.array(points[mask])
    
    return valid_points


def calcHomography(Hmatrix,point,inverse=False):
    
    matrix = Hmatrix
    auxPoint = point
    if inverse:
        matrix = np.linalg.inv(Hmatrix)

    if len(point)==2:
        auxPoint = np.append(point,1)

    hpoint = np.matmul(matrix,auxPoint)

    #Divide for Z coordinate to get form [X,Y,1]
    hpoint /= hpoint[-1]

    return hpoint[:2]


def applyHom(img,h_points,H_X_axis,H_Y_axis,img_X_axis,img_Y_axis,Hmatrix,one2oneMaping = False, singleDimension = False):

    newImg = np.zeros((len(H_Y_axis),len(H_X_axis),3),dtype='float64')
    img_points = np.zeros(h_points.shape,dtype='int64')

    HmatrixI = np.linalg.inv(Hmatrix)

    h_points_X = h_points[:,0]
    h_points_Y = h_points[:,1]
    # img_points_X = h_points[:,0]
    # h_points_Y = h_points[:,1]

    h_points_XI = h_points_X*HmatrixI[0,0]+ h_points_Y*HmatrixI[0,1] + HmatrixI[0,2] 
    h_points_YI = h_points_X*HmatrixI[1,0]+ h_points_Y*HmatrixI[1,1] + HmatrixI[1,2] 
    k = h_points_X*HmatrixI[2,0]+ h_points_Y*HmatrixI[2,1] + HmatrixI[2,2] 

    h_points_XI = h_points_XI / k
    h_points_YI = h_points_YI / k

    #Denormalization on img space
    step_x = abs(img_X_axis[0] - img_X_axis[1])
    min_val_x = img_X_axis[0]
    step_y = abs(img_Y_axis[0] - img_Y_axis[1])
    min_val_y = img_Y_axis[-1]

    denor_points_X = np.around((h_points_XI - min_val_x)/step_x)
    denor_points_Y = np.around((h_points_YI - min_val_y)/step_y)

    #Denormalization on hom space
    step_x = abs(H_X_axis[0] - H_X_axis[1])
    min_val_x = H_X_axis[0]
    step_y = abs(H_Y_axis[0] - H_Y_axis[1])
    min_val_y = H_Y_axis[-1]

    denor_h_points_X = np.around((h_points_X - min_val_x)/step_x)
    denor_h_points_Y = np.around((h_points_Y - min_val_y)/step_y)

    # Filtering edge points
    mask = denor_points_X < img.shape[1] 
    denor_points_X = denor_points_X[mask]
    denor_points_Y = denor_points_Y[mask]
    denor_h_points_X = denor_h_points_X[mask]
    denor_h_points_Y = denor_h_points_Y[mask]
    h_points = h_points[mask]
    img_points = img_points[mask]

    
    mask = denor_points_Y < img.shape[0]
    denor_points_X = denor_points_X[mask]
    denor_points_Y = denor_points_Y[mask]
    denor_h_points_X = denor_h_points_X[mask]
    denor_h_points_Y = denor_h_points_Y[mask]
    h_points = h_points[mask]
    img_points = img_points[mask]

    mask = denor_points_X > 0 
    denor_points_X = denor_points_X[mask]
    denor_points_Y = denor_points_Y[mask]
    denor_h_points_X = denor_h_points_X[mask]
    denor_h_points_Y = denor_h_points_Y[mask]
    h_points = h_points[mask]
    img_points = img_points[mask]

    
    mask = denor_points_Y > 0
    denor_points_X = denor_points_X[mask]
    denor_points_Y = denor_points_Y[mask]
    denor_h_points_X = denor_h_points_X[mask]
    denor_h_points_Y = denor_h_points_Y[mask]
    h_points = h_points[mask]
    img_points = img_points[mask]

    #Paring points
    h_points[:,1]= denor_h_points_X
    h_points[:,0]= denor_h_points_Y
    h_points = h_points.astype(int)
    
    img_points[:,1]= denor_points_X
    img_points[:,0]= denor_points_Y
    img_points = img_points.astype(int)

    #Setting corresponding values for images
    conv_img = img

    if (not one2oneMaping):
        kernel = np.ones((3,3))/9
        conv_img = cv.filter2D(img,-1,kernel=kernel)

    if (singleDimension):
        conv_img = np.reshape(conv_img,(conv_img.shape[0],conv_img.shape[1],-1))
    

    for i in range(len(h_points)):

        newImg[h_points[i][0],h_points[i][1]] = conv_img[img_points[i][0],img_points[i][1]]

    return newImg

def getSquarePoint(center,img,singleDimension=False):

    X_start = center[0]-1
    X_end = center[0]+1

    Y_start = center[1]-1
    Y_end = center[1]+1

    if X_start < 0: 
        X_start = 0
    if X_end > img.shape[1]-1:
        X_end = img.shape[1]-1
    
    if Y_start < 0: 
        Y_start = 0
    if Y_end > img.shape[0]-1:
        Y_end = img.shape[0]-1

    X_value = np.mean( img[Y_start:Y_end,X_start:X_end,0])

    if (singleDimension):
        return np.array( [X_value])

    Y_value = np.mean( img[Y_start:Y_end,X_start:X_end,1])
    Z_value = np.mean( img[Y_start:Y_end,X_start:X_end,2])

    return np.array((X_value,Y_value,Z_value))


def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))
