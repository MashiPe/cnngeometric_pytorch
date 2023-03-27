from __future__ import print_function, division
import os
import sys
from skimage import io
import pandas as pd
import numpy as np
import torch
from torch.nn.modules.module import Module
from torch.utils.data import Dataset
from torch.autograd import Variable
import torch.nn.functional as F
from util.torch_util import expand_dim
# from stitching.stitch import extendAxis

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

def genFlexPoints(img_h=240,img_w=240):

    # create grid in numpy
    # self.grid = np.zeros( [self.out_h, self.out_w, 3], dtype=np.float32)
    # sampling grid with dim-0 coords (Y)

    x_axis = np.linspace(-1,1,img_w)
    y_axis = np.linspace(-1,1,img_h)

    x_axis = extendAxis(x_axis)
    y_axis = extendAxis(y_axis)

    grid_X,grid_Y = np.meshgrid(x_axis,y_axis)
    # grid_X,grid_Y: size [1,H,W,1,1]
    grid_X = torch.FloatTensor(grid_X).unsqueeze(0).unsqueeze(3)
    grid_Y = torch.FloatTensor(grid_Y).unsqueeze(0).unsqueeze(3)

    return grid_X,grid_Y

    
class StitchingAffineGridGen(Module):
    def __init__(self, img_h=240, img_w=240 ,out_h=480, out_w=480, use_cuda=True):
        super(StitchingAffineGridGen, self).__init__()        
        self.out_h, self.out_w = out_h, out_w
        self.img_h, self.img_w = img_h, img_w
        self.use_cuda = use_cuda

        # grid_X,grid_Y = genFlexPoints(img_h,img_w)
        self.grid_X,self.grid_Y = np.meshgrid(np.linspace(-1,1,out_w),np.linspace(-1,1,out_h))
        # grid_X,grid_Y: size [1,H,W,1,1]
        self.grid_X = torch.FloatTensor(self.grid_X).unsqueeze(0).unsqueeze(3)
        self.grid_Y = torch.FloatTensor(self.grid_Y).unsqueeze(0).unsqueeze(3)
        
        self.grid_X = Variable(self.grid_X,requires_grad=False)
        self.grid_Y = Variable(self.grid_Y,requires_grad=False)
        if use_cuda:
            self.grid_X = self.grid_X.cuda()
            self.grid_Y = self.grid_Y.cuda()
            
    def forward(self, theta,custom_grid=False,custom_grid_X = None, custom_grid_Y = None):
        b=theta.size(0)
        if not theta.size()==(b,6):
            theta = theta.view(b,6)
            theta = theta.contiguous()
            
        t0=theta[:,0].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        t1=theta[:,1].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        t2=theta[:,2].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        t3=theta[:,3].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        t4=theta[:,4].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        t5=theta[:,5].unsqueeze(1).unsqueeze(2).unsqueeze(3)

        grid_X = custom_grid_X
        grid_Y = custom_grid_Y

        if not custom_grid:
            grid_X = expand_dim(self.grid_X,0,b)
            grid_Y = expand_dim(self.grid_Y,0,b)

        grid_Xp = grid_X*t0 + grid_Y*t1 + t2
        grid_Yp = grid_X*t3 + grid_Y*t4 + t5
        
        return torch.cat((grid_Xp,grid_Yp),3)
 
    def inverse(self, theta,custom_grid=False,custom_grid_X = None, custom_grid_Y = None):
        b=theta.size(0)
        if not theta.size()==(b,6):
            theta = theta.view(b,6)
            theta = theta.contiguous()
            

        t2=theta[:,2].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        t5=theta[:,5].unsqueeze(1).unsqueeze(2).unsqueeze(3)

        grid_X = custom_grid_X
        grid_Y = custom_grid_Y

        if not custom_grid:
            grid_X = expand_dim(self.grid_X,0,b)
            grid_Y = expand_dim(self.grid_Y,0,b)

        # aff_matrix = np.array( [[t0.detach().cpu().numpy()[0,0,0,0],t1.detach().cpu().numpy()[0,0,0,0]],[t3.detach().cpu().numpy()[0,0,0,0],t4.detach().cpu().numpy()[0,0,0,0]]] )

        aff_matrix = torch.cat((theta[:,:2],theta[:,3:5]),axis=1)
        aff_matrix = torch.reshape(aff_matrix,(2,2))

        aff_matrix_i = torch.linalg.inv(aff_matrix) 


        t0=aff_matrix_i[0,0].unsqueeze(0).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        t1=aff_matrix_i[0,1].unsqueeze(0).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        t3=aff_matrix_i[1,0].unsqueeze(0).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        t4=aff_matrix_i[1,1].unsqueeze(0).unsqueeze(1).unsqueeze(2).unsqueeze(3)

        grid_X = grid_X - t2
        grid_Y = grid_Y - t5
        
        grid_Xp = grid_X*t0 + grid_Y*t1
        grid_Yp = grid_X*t3 + grid_Y*t4
        
        return torch.cat((grid_Xp,grid_Yp),3)
    
    def inversev2(self, theta,custom_grid=False,custom_grid_X = None, custom_grid_Y = None):
        
        grid = self.forward(theta)

        aux_grid = torch.full(grid.size(),-2)

        grid_x = grid[:,:,:,0]
        grid_y = grid[:,:,:,1]
        grid_x = torch.mul(torch.add(grid_x,1),self.img_w/2)
        grid_y = torch.mul(torch.add(grid_y,1),self.img_h/2)
        
        for b in range(grid.size(0)):
            for h in range(grid.size(1)):
                for w in range(grid.size(2)):
                    aux_x = int(grid_x[b,h,w].detach().cpu().numpy())
                    aux_y = int(grid_y[b,h,w].detach().cpu().numpy())

                    if aux_x >= 0 and aux_x<grid.size(2) and aux_y>=0 and aux_y<grid.size(1):
                        aux_grid[b,aux_y,aux_x,0] = torch.FloatTensor([((w / self.img_w)*2)-1])
                        aux_grid[b,aux_y,aux_x,1] = torch.FloatTensor([((h/self.img_h)*2)-1])

        aux_grid = aux_grid.type(torch.FloatTensor)

        if (self.use_cuda):
            aux_grid = aux_grid.cuda()

        return aux_grid

    

class StitchingHomographyGridGen(Module):
    def __init__(self, img_h=240 , img_w=240 ,out_h=480, out_w=480, use_cuda=True):
        super(StitchingHomographyGridGen, self).__init__()        
        self.out_h, self.out_w = out_h, out_w
        self.img_h, self.img_w = img_h, img_w
        self.use_cuda = use_cuda

        grid_X,grid_Y = genFlexPoints(img_h,img_w)
        
        self.grid_X = Variable(grid_X,requires_grad=False)
        self.grid_Y = Variable(grid_Y,requires_grad=False)
        if use_cuda:
            self.grid_X = self.grid_X.cuda()
            self.grid_Y = self.grid_Y.cuda()
            
    def forward(self, theta,custom_grid=False,custom_grid_X = None, custom_grid_Y = None):
        b=theta.size(0)
        # b=1
        if theta.size(1)==9:
            H = theta            
        else:
            H = homography_mat_from_4_pts(theta)            
        h0=H[:,0].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        h1=H[:,1].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        h2=H[:,2].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        h3=H[:,3].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        h4=H[:,4].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        h5=H[:,5].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        h6=H[:,6].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        h7=H[:,7].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        h8=H[:,8].unsqueeze(1).unsqueeze(2).unsqueeze(3)

        grid_X = custom_grid_X
        grid_Y = custom_grid_Y

        if not custom_grid:
            grid_X = expand_dim(self.grid_X,0,b);
            grid_Y = expand_dim(self.grid_Y,0,b);


        grid_Xp = grid_X*h0+grid_Y*h1+h2
        grid_Yp = grid_X*h3+grid_Y*h4+h5
        k = grid_X*h6+grid_Y*h7+h8

        grid_Xp /= k; grid_Yp /= k
        
        return torch.cat((grid_Xp,grid_Yp),3)

    def inverse(self, theta,custom_grid=False,custom_grid_X = None, custom_grid_Y = None):
        b=theta.size(0)
        # b=1
        if theta.size(1)==9:
            H = theta            
        else:
            H = homography_mat_from_4_pts(theta)            
        
        H = torch.reshape(H,(3,3))        
        H = torch.transpose(H,0,1)
        H = torch.linalg.inv(H)
        H = torch.transpose(H,0,1)
        H = torch.reshape(H,(1,9))

        h0=H[:,0].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        h1=H[:,1].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        h2=H[:,2].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        h3=H[:,3].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        h4=H[:,4].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        h5=H[:,5].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        h6=H[:,6].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        h7=H[:,7].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        h8=H[:,8].unsqueeze(1).unsqueeze(2).unsqueeze(3)

        grid_X = custom_grid_X
        grid_Y = custom_grid_Y

        if not custom_grid:
            grid_X = expand_dim(self.grid_X,0,b);
            grid_Y = expand_dim(self.grid_Y,0,b);

        grid_Xp = grid_X*h0+grid_Y*h1+h2
        grid_Yp = grid_X*h3+grid_Y*h4+h5
        k = grid_X*h6+grid_Y*h7+h8

        grid_Xp /= k; grid_Yp /= k
        
        return torch.cat((grid_Xp,grid_Yp),3)

def homography_mat_from_4_pts(theta):
    b=theta.size(0)
    if not theta.size()==(b,8):
        theta = theta.view(b,8)
        theta = theta.contiguous()
    
    xp=theta[:,:4].unsqueeze(2) ;yp=theta[:,4:].unsqueeze(2) 
    
    x = Variable(torch.FloatTensor([-1, -1, 1, 1])).unsqueeze(1).unsqueeze(0).expand(b,4,1)
    y = Variable(torch.FloatTensor([-1,  1,-1, 1])).unsqueeze(1).unsqueeze(0).expand(b,4,1)
    z = Variable(torch.zeros(4)).unsqueeze(1).unsqueeze(0).expand(b,4,1)
    o = Variable(torch.ones(4)).unsqueeze(1).unsqueeze(0).expand(b,4,1)
    single_o = Variable(torch.ones(1)).unsqueeze(1).unsqueeze(0).expand(b,1,1)
    
    if theta.is_cuda:
        x = x.cuda()
        y = y.cuda()
        z = z.cuda()
        o = o.cuda()
        single_o = single_o.cuda()


    A=torch.cat([torch.cat([-x,-y,-o,z,z,z,x*xp,y*xp,xp],2),torch.cat([z,z,z,-x,-y,-o,x*yp,y*yp,yp],2)],1)
    # find homography by assuming h33 = 1 and inverting the linear system
    h=torch.bmm(torch.inverse(A[:,:,:8]),-A[:,:,8].unsqueeze(2))
    # add h33
    h=torch.cat([h,single_o],1)
    
    H = h.squeeze(2)

    # print("Homografy Matrix")
    # print(H.size())
    # print(H)
    
    return H

class StitchingTpsGridGen(Module):
    def __init__(self, img_h=240, img_w=240, use_regular_grid=True, x_axis_coords=None,y_axis_coords=None , reg_factor=0, use_cuda=True):
        super(StitchingTpsGridGen, self).__init__()
        # self.out_h, self.out_w = out_h, out_w
        self.reg_factor = reg_factor
        self.img_h, self.img_w = img_h, img_w
        self.use_cuda = use_cuda
        self.regular_grid = use_regular_grid
        self.use_cuda = use_cuda

        grid_X,grid_Y = genFlexPoints(img_h,img_w)
        
        self.grid_X = Variable(grid_X,requires_grad=False)
        self.grid_Y = Variable(grid_Y,requires_grad=False)
        if use_cuda:
            self.grid_X = self.grid_X.cuda()
            self.grid_Y = self.grid_Y.cuda()

        # initialize regular grid for control points P_i
        if use_regular_grid:
            grid_size = 3
            axis_coords = np.linspace(-1,1,grid_size)
            self.N = grid_size*grid_size
            P_Y,P_X = np.meshgrid(axis_coords,axis_coords)
            self.x_axis_coords = P_X
            self.y_axis_coords = P_Y
            P_X = np.reshape(P_X,(-1,1)) # size (N,1)
            P_Y = np.reshape(P_Y,(-1,1)) # size (N,1)
            P_X = torch.FloatTensor(P_X)
            P_Y = torch.FloatTensor(P_Y)
        else:
            self.N = len(x_axis_coords) 
            # P_Y,P_X = np.meshgrid(y_axis_coords,x_axis_coords)
            # P_X = np.reshape(P_X,(-1,1)) # size (N,1)
            # P_Y = np.reshape(P_Y,(-1,1)) # size (N,1)
            # P_X = np.reshape(np.delete(P_X,[4]),(-1,1))
            # P_Y = np.reshape(np.delete(P_Y,[4]),(-1,1))
            self.x_axis_coords = x_axis_coords
            self.y_axis_coords = y_axis_coords
            x_axis_coords = np.reshape(x_axis_coords,(-1,1))
            y_axis_coords = np.reshape(y_axis_coords,(-1,1))
            P_X = torch.FloatTensor(x_axis_coords)
            P_Y = torch.FloatTensor(y_axis_coords)

        self.Li = Variable(self.compute_L_inverse(P_X,P_Y).unsqueeze(0),requires_grad=False)
        self.P_X = P_X.unsqueeze(2).unsqueeze(3).unsqueeze(4).transpose(0,4)
        self.P_Y = P_Y.unsqueeze(2).unsqueeze(3).unsqueeze(4).transpose(0,4)
        self.P_X = Variable(self.P_X,requires_grad=False)
        self.P_Y = Variable(self.P_Y,requires_grad=False)
        if use_cuda:
            self.P_X = self.P_X.cuda()
            self.P_Y = self.P_Y.cuda()

            
    def forward(self, theta,custom_grid=False,custom_grid_X = None, custom_grid_Y = None):
        grid_X = custom_grid_X
        grid_Y = custom_grid_Y
        if not custom_grid:
            grid_X = self.grid_X
            grid_Y = self.grid_Y
            
        warped_grid = self.apply_transformation(theta,torch.cat((grid_X,grid_Y),3))
        
        return warped_grid
    
    def compute_L_inverse(self,X,Y):
        N = X.size()[0] # num of points (along dim 0)
        # construct matrix K
        Xmat = X.expand(N,N)
        Ymat = Y.expand(N,N)
        P_dist_squared = torch.pow(Xmat-Xmat.transpose(0,1),2)+torch.pow(Ymat-Ymat.transpose(0,1),2)
        P_dist_squared[P_dist_squared==0]=1 # make diagonal 1 to avoid NaN in log computation
        K = torch.mul(P_dist_squared,torch.log(P_dist_squared))
        if self.reg_factor != 0:
            K+=torch.eye(K.size(0),K.size(1))*self.reg_factor
        # construct matrix L
        O = torch.FloatTensor(N,1).fill_(1)
        Z = torch.FloatTensor(3,3).fill_(0)       
        P = torch.cat((O,X,Y),1)
        L = torch.cat((torch.cat((K,P),1),torch.cat((P.transpose(0,1),Z),1)),0)
        Li = torch.inverse(L)
        if self.use_cuda:
            Li = Li.cuda()
        return Li
        
    def apply_transformation(self,theta,points):
        if theta.dim()==2:
            theta = theta.unsqueeze(2).unsqueeze(3)
        # points should be in the [B,H,W,2] format,
        # where points[:,:,:,0] are the X coords  
        # and points[:,:,:,1] are the Y coords  
        
        # input are the corresponding control points P_i
        batch_size = theta.size()[0]
        # split theta into point coordinates
        Q_X=theta[:,:self.N,:,:].squeeze(3)
        Q_Y=theta[:,self.N:,:,:].squeeze(3)
        
        # get spatial dimensions of points
        points_b = points.size()[0]
        points_h = points.size()[1]
        points_w = points.size()[2]
        
        # repeat pre-defined control points along spatial dimensions of points to be transformed
        P_X = self.P_X.expand((1,points_h,points_w,1,self.N))
        P_Y = self.P_Y.expand((1,points_h,points_w,1,self.N))
        
        # compute weigths for non-linear part
        W_X = torch.bmm(self.Li[:,:self.N,:self.N].expand((batch_size,self.N,self.N)),Q_X)
        W_Y = torch.bmm(self.Li[:,:self.N,:self.N].expand((batch_size,self.N,self.N)),Q_Y)
        # reshape
        # W_X,W,Y: size [B,H,W,1,N]
        W_X = W_X.unsqueeze(3).unsqueeze(4).transpose(1,4).repeat(1,points_h,points_w,1,1)
        W_Y = W_Y.unsqueeze(3).unsqueeze(4).transpose(1,4).repeat(1,points_h,points_w,1,1)
        # compute weights for affine part
        A_X = torch.bmm(self.Li[:,self.N:,:self.N].expand((batch_size,3,self.N)),Q_X)
        A_Y = torch.bmm(self.Li[:,self.N:,:self.N].expand((batch_size,3,self.N)),Q_Y)
        # reshape
        # A_X,A,Y: size [B,H,W,1,3]
        A_X = A_X.unsqueeze(3).unsqueeze(4).transpose(1,4).repeat(1,points_h,points_w,1,1)
        A_Y = A_Y.unsqueeze(3).unsqueeze(4).transpose(1,4).repeat(1,points_h,points_w,1,1)
        
        # compute distance P_i - (grid_X,grid_Y)
        # grid is expanded in point dim 4, but not in batch dim 0, as points P_X,P_Y are fixed for all batch
        points_X_for_summation = points[:,:,:,0].unsqueeze(3).unsqueeze(4).expand(points[:,:,:,0].size()+(1,self.N))
        points_Y_for_summation = points[:,:,:,1].unsqueeze(3).unsqueeze(4).expand(points[:,:,:,1].size()+(1,self.N))
        
        if points_b==1:
            delta_X = points_X_for_summation-P_X
            delta_Y = points_Y_for_summation-P_Y
        else:
            # use expanded P_X,P_Y in batch dimension
            delta_X = points_X_for_summation-P_X.expand_as(points_X_for_summation)
            delta_Y = points_Y_for_summation-P_Y.expand_as(points_Y_for_summation)
            
        dist_squared = torch.pow(delta_X,2)+torch.pow(delta_Y,2)
        # U: size [1,H,W,1,N]
        dist_squared[dist_squared==0]=1 # avoid NaN in log computation
        U = torch.mul(dist_squared,torch.log(dist_squared)) 
        
        # expand grid in batch dimension if necessary
        points_X_batch = points[:,:,:,0].unsqueeze(3)
        points_Y_batch = points[:,:,:,1].unsqueeze(3)
        if points_b==1:
            points_X_batch = points_X_batch.expand((batch_size,)+points_X_batch.size()[1:])
            points_Y_batch = points_Y_batch.expand((batch_size,)+points_Y_batch.size()[1:])
        
        points_X_prime = A_X[:,:,:,:,0]+ \
                       torch.mul(A_X[:,:,:,:,1],points_X_batch) + \
                       torch.mul(A_X[:,:,:,:,2],points_Y_batch) + \
                       torch.sum(torch.mul(W_X,U.expand_as(W_X)),4)
                    
        points_Y_prime = A_Y[:,:,:,:,0]+ \
                       torch.mul(A_Y[:,:,:,:,1],points_X_batch) + \
                       torch.mul(A_Y[:,:,:,:,2],points_Y_batch) + \
                       torch.sum(torch.mul(W_Y,U.expand_as(W_Y)),4)
        
        return torch.cat((points_X_prime,points_Y_prime),3)

    def inverse(self, theta,custom_grid=False,custom_grid_X = None, custom_grid_Y = None):
        
        auxP_X = self.P_X
        auxP_Y = self.P_Y
        
        theta_np = theta[0].detach().cpu().numpy()

        x_axis_coords = theta_np[:theta_np.shape[0]//2]
        y_axis_coords = theta_np[theta_np.shape[0]//2:]

        #Recalculate grid to do transformation
        if self.regular_grid:
            grid_size = 3
            axis_coords = np.linspace(-1,1,grid_size)
            self.N = grid_size*grid_size
            P_Y,P_X = np.meshgrid(axis_coords,axis_coords)
            P_X = np.reshape(P_X,(-1,1)) # size (N,1)
            P_Y = np.reshape(P_Y,(-1,1)) # size (N,1)
            P_X = torch.FloatTensor(P_X)
            P_Y = torch.FloatTensor(P_Y)
        else:
            self.N = len(x_axis_coords) 
            # P_Y,P_X = np.meshgrid(y_axis_coords,x_axis_coords)
            # P_X = np.reshape(P_X,(-1,1)) # size (N,1)
            # P_Y = np.reshape(P_Y,(-1,1)) # size (N,1)
            # P_X = np.reshape(np.delete(P_X,[4]),(-1,1))
            # P_Y = np.reshape(np.delete(P_Y,[4]),(-1,1))
            x_axis_coords = np.reshape(x_axis_coords,(-1,1))
            y_axis_coords = np.reshape(y_axis_coords,(-1,1))
            P_X = torch.FloatTensor(x_axis_coords)
            P_Y = torch.FloatTensor(y_axis_coords)

        self.Li = Variable(self.compute_L_inverse(P_X,P_Y).unsqueeze(0),requires_grad=False)
        self.P_X = P_X.unsqueeze(2).unsqueeze(3).unsqueeze(4).transpose(0,4)
        self.P_Y = P_Y.unsqueeze(2).unsqueeze(3).unsqueeze(4).transpose(0,4)
        self.P_X = Variable(self.P_X,requires_grad=False)
        self.P_Y = Variable(self.P_Y,requires_grad=False)

        if self.use_cuda:
            self.P_X = self.P_X.cuda()
            self.P_Y = self.P_Y.cuda()


        grid_X = custom_grid_X
        grid_Y = custom_grid_Y
        if not custom_grid:
            grid_X = self.grid_X
            grid_Y = self.grid_Y


        aux_theta = torch.FloatTensor(np.expand_dims(np.concatenate((self.x_axis_coords,self.x_axis_coords)),axis=0))   

        if self.use_cuda:
            aux_theta = aux_theta.cuda()
            
        warped_grid = self.apply_transformation(aux_theta,torch.cat((grid_X,grid_Y),3))

        self.P_X = auxP_X
        self.P_Y = auxP_Y
        
        return warped_grid