import pyopencl as cl
import numpy as np


class PolyV2:

    kernel_src = """
                    __kernel void check( float *a,  float *b, float *c , int *dir 
                                        , __global float *Px , __global float *Py, __global int *mask){
                        int gid = get_global_id(0);
                        mask[gid] = 0;
                        
                        int aux = 0;

                        for(int k=0; k<4; k++)
                        {
                            fun_res = (a[k]*Px[gid])+(b[k]*Py[gid])+c[k];

                            if (dir[k]==0){
                                if( fun_res>0 ){
                                    aux++;
                                }
                            }else{
                                if( fun_res<0 ){
                                    aux++;
                                }
                            }
                        }

                        if (aux == 4 ){
                            mask[gid] = 1;
                        }
                    }
                 """

    def __init__(self,point_list,ref_point) :
        
        self.ref_point = ref_point
        self.list_a = []
        self.list_b = []
        self.list_c = []
        self.list_direction = []

        for i in range(len(point_list)):
            
            point_a = point_list[i]

            b_index = i+1
            
            if (b_index == len(point_list)):
                b_index = 0

            point_b = point_list[b_index]

            a,b,c,d = self._genInecParams(point_a,point_b)

            self.list_a.append(a)
            self.list_b.append(b)
            self.list_c.append(c)
            self.list_direction.append(d)

        self.list_a = np.array(self.list_a,dtype=np.float32) 
        self.list_b = np.array(self.list_b,dtype=np.float32) 
        self.list_c = np.array(self.list_c,dtype=np.float32) 
        self.list_direction = np.array(self.list_direction,dtype=np.int32)

        # self.ctx = cl.create_some_context()
        # self.queue = cl.CommandQueue(self.ctx)

        # self.prg = cl.Program(self.ctx,self.kernel_src).build()


    def _genInecParams(self,A,B):
        
        # Line AB represented as a1x + b1y +c1 = 0
        a1 = A[1] - B[1]
        b1 = B[0] - A[0]
        c1 = -1*(a1*A[0] + b1*A[0])

        aux = (a1*self.ref_point[0])+(b1*self.ref_point[1])+c1

        d=1

        if aux > 0:
            d=0
        
        return a1,b1,c1,d

    def containsPointsCL(self,points):

        mf = cl.mem_flags

        a_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.list_a)
        b_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.list_b)
        c_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.list_c)
        d_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.list_direction)
        
        Px = points[:,0]
        Py = points[:,1]
        mask = np.zeros((len(points)),dtype=np.int32)

        Px_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=Px)
        Py_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=Py)
        mask_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, mask.nbytes)

        self.prg.check(self.queue,mask.shape,None,a_buf,b_buf,c_buf,d_buf,
                            Px_buf,Py_buf,mask_buf)

        res_mask = np.empty_like(mask)

        cl.enqueue_copy(self.queue,res_mask,mask_buf)

        res_mask = [ x==1 for x in res_mask ]

        return res_mask
    
    def containsPoints(self,points):
        
        Px = points[:,0]
        Py = points[:,1]

        for i in range(len(self.list_a)):

            aux_Px = Px * self.list_a[i]
            aux_Py = Py * self.list_b[i]

            aux_res = aux_Px + aux_Py + self.list_c[i]

            if ( self.list_direction[i] == 0 ):
                mask = aux_res > 0
            else:
                mask = aux_res < 0

            Px = Px[mask]
            Py = Py[mask]

        valid_points = np.empty((len(Px),2))
        valid_points[:,0] = Px
        valid_points[:,1] = Py

        return valid_points