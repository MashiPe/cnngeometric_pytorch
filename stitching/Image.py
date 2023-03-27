import numpy as np
import torch
import cv2 as cv
from typing import Any, List

class Image:

    def __init__(self,path:str,
                index:int,
                transform_params_stack: List[torch.Tensor] = None,
                resize = False , 
                out_h = 240, 
                out_w = 240,
                grid = None ):
        
        img = cv.imread(path)

        if resize:        
            img = cv.resize(img,(out_w,out_h))
        
        self.img = img 
        self.transform_stack  = transform_params_stack 
        self.index = index
        self.grid = None 