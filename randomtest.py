
import cv2 as cv
import util.blend_util as blend
import numpy as np

coords_1 = np.array([[0,0],[0,1],[1,0],[1,1]])
coords_2 = np.array([[0,2],[0,3],[1,2],[1,3]])

weights = cv.imread("./ferret.jpg")

# weights = blend.genWieghts(240,250)

# result = np.reshape(cv.filter2D(weights,-1,kernel=kernel),(weights.shape[0],weights.shape[1],-1))
# result = cv.filter2D(weights,-1,kernel=kernel)

weights[(0,0)] = weights[(0,1)]

print(weights)