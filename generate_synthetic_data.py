import numpy as np
def banded_matrix(dimension):
    x_true=np.eye(dimension)
    for i in range(len(x_true)):
        for j in range(len(x_true)):
            if abs(i-j)<=10:   
                x_true[i][j]=1-abs(i-j)/10
    return x_true
def block_matrix(dimension,group):
    x_true=np.eye(dimension)
    part=int(dimension/group)
    x_submatrix=np.eye(part)

    for i in range (part):
        for j in range(part):
            if i!=j:
                x_submatrix[i][j]=0.6
    for k in range(0,group):
        x_true[k*part:(k+1)*part,k*part:(k+1)*part]=x_submatrix
    return x_true
def toeplitz_matrix(dimension):
    x_true=np.eye(dimension)
    for i in range(len(x_true)):
        for j in range(len(x_true)):               
            x_true[i][j]=0.75**abs(i-j)
    return x_true

def generate_scm(x_true,sample_size):
    mu=np.zeros(len(x_true))
    data_sample=np.random.multivariate_normal(mean=mu,cov=x_true,size=sample_size)
    sample_vector=data_sample-np.mean(data_sample,axis=0)
    s=np.dot(sample_vector.T,sample_vector)/(sample_size-1)
    return s