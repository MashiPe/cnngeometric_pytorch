

class Poly:

    def __init__(self,point_list,ref_point) -> None:
        
        self.inecs = []
        self.ref_point = ref_point

        for i in range(len(point_list)):
            
            point_a = point_list[i]

            b_index = i+1
            
            if (b_index == len(point_list)):
                b_index = 0

            point_b = point_list[b_index]

            self.inecs.append(self._genInec(point_a,point_b))
            
    def _leftInec(self,a1,b1,c1,point):
        aux = (a1*point[0]+b1*point[1]+c1)
        res = aux > 0
        return res
    
    def _rightInec(self,a1,b1,c1,point):
        aux = (a1*point[0]+b1*point[1]+c1)
        res = aux < 0
        return res
    
    def _genInec(self,A,B):
        
        # Line AB represented as a1x + b1y +c1 = 0
        a1 = A[1] - B[1]
        b1 = B[0] - A[0]
        c1 = -1*(a1*A[0] + b1*A[0])


        if (self._rightInec(a1,b1,c1,self.ref_point)):
            return (a1,b1,c1,1,self._rightInec)
        
        return (a1,b1,c1,0,self._leftInec)

    def contains(self,point):

        for inec in self.inecs:
            if (not inec[3](inec[0],inec[1],inec[2],point)):
                return False
        
        return True