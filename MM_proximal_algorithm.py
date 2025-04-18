import numpy as np 
class PDTE_FC(object):
    def __init__(self,x,s,t,fhi,gamma,lambda_val,a_val,dimension):
        self.x=x
        self.s=s #sampe covariance
        self.t=t #tau
        self.fhi=fhi #step size 
        self.gamma=gamma
        self.lambda_val=lambda_val
        self.a_val=a_val
        self.dimension=dimension
    def f_function(self,x):
        return 0.5*(np.linalg.norm(x-self.s)**2)-self.t*np.log(np.linalg.det(x))
    def gradient(self,x):
        inv=np.linalg.inv(x)
        inv_x=0.5*(inv+inv.T)
        return x-self.s-self.t*inv_x
    def softthresholding(self,b, lam):
        soft_thresh=np.zeros(b.shape)
        for i in range(len(b)):
            for j in range((len(b))):
                ele=b[i][j]
                soft_thresh[i][j] = np.sign(ele) * max(abs(ele) - lam[i][j], 0) 
        return soft_thresh   
    def mcp_derivative(self,x):
        
        return (self.lambda_val-abs(x)/self.a_val)*(abs(x)<=self.lambda_val*self.a_val)+0*(abs(x)>self.lambda_val*self.a_val)    
    def mcp_penalty(self, x):
        is_linear = (np.abs(x) <= self.lambda_val*self.a_val)
        # is_quadratic = np.logical_and(self.lambda_val < np.abs(x), np.abs(x) <= self.a_val * self.lambda_val)
        is_constant = (self.a_val * self.lambda_val) < np.abs(x)
        
        linear_part = (self.lambda_val*abs(x)-(x**2)/(2*self.a_val) )* is_linear
        # quadratic_part = (2 * self.a_val * self.lambda_val * np.abs(x) - abs(x)**2 - self.lambda_val**2) / (2 * (self.a_val - 1)) * is_quadratic
        constant_part =  (0.5*(self.lambda_val**2)*self.a_val)* is_constant
        return linear_part + constant_part
    def update_w(self,x):
        scad_matrix=self.mcp_derivative(x)
        return scad_matrix-np.diag(scad_matrix.diagonal())
    
    def function_value(self,x):
        return (1/2)*np.linalg.norm(x-self.s,'fro')**2-self.t*np.log(np.linalg.det(x))-np.trace(self.mcp_penalty(x))+np.sum(self.mcp_penalty(x))    
                    
    def process(self,x):
        value_list=[]   
        x=self.x
        for iteration in range(0,100):
            iteration=0
            flag1=True
            x_outer_old=x
            w=self.update_w(x_outer_old)
            value_list_inner=[]
            if iteration==0:    
                value_list_inner.append((1/2)*np.linalg.norm(self.x-self.s,'fro')**2-self.t*np.log(np.linalg.det(self.x))+np.sum(abs(self.x)*w))
            while flag1:
                flag2=True
                iteration=iteration+1
                x_old=x
                if iteration==1:
                    fhi_t=self.fhi
                else:
                    fhi_t=max(self.fhi, (1/self.gamma)*fhi_t)
                while flag2:
                    x=x_old-(1/fhi_t)*self.gradient(x_old)
                    x=self.softthresholding(x,w/fhi_t)
                    a1=self.f_function(x_old)
                    a2=np.sum(self.gradient(x_old)*(x-x_old))
                    a3=np.linalg.norm(x-x_old)**2
                    g_value=a1+a2+0.5*fhi_t*a3
                    f_value=self.f_function(x)
                    if g_value>=f_value and min(np.linalg.eigh(x)[0])>0 :
                        x_new=x
                        break
                    else:
                        fhi_t=self.gamma*fhi_t
                value_list.append(self.function_value(x_new))
                if np.linalg.norm(x_new-x_old)/np.linalg.norm(x_old)<1e-5:
                    break
            if np.linalg.norm(x_new-x_outer_old)/np.linalg.norm(x_outer_old)<1e-5:
                print('PDTE_FC收敛')
                return x_new
        return x_new
                    
                    
                
            
    
  
                    
                
            
    
    
