import math
import numpy as np
from shapely.geometry import Polygon,Point
import scipy.ndimage as ndimage
from sklearn.preprocessing import normalize

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

# Takes coordinate points from a numpy array and returns points that are contained within 
# the bounding box.
def getImgPoints(boundBox, points):

    poly = Polygon(boundBox)

    valid_points = np.array([ point for point in points if poly.contains(Point(point))])
    
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

    newImg = np.zeros((len(H_Y_axis),len(H_X_axis),3))

    for point in h_points:
        inverse_point = calcHomography(Hmatrix,point,inverse=True)

        #Calc the nearest point in the axis from the result of the inverse homography
        inverse_point_x = np.array([min(img_X_axis, key=lambda x: abs(x-inverse_point[0]))])   
        inverse_point_y = np.array([min(img_Y_axis, key=lambda y: abs(y-inverse_point[1]))])

        #Get denormalized position from axis points
        real_Hpoint_x = np.where(H_X_axis == point[0])[0][0]
        real_Hpoint_y = np.where(H_Y_axis == point[1])[0][0]

        real_point_x = np.where(img_X_axis == inverse_point_x[0])[0][0]
        real_point_y = np.where(img_Y_axis == inverse_point_y[0])[0][0]

        if (one2oneMaping) :
            newImg[real_Hpoint_y,real_Hpoint_x] = img[real_point_y,real_point_x]
        else:
            newImg[real_Hpoint_y,real_Hpoint_x] = getSquarePoint((real_point_x,real_point_y),img,singleDimension)
    
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
