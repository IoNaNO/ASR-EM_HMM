import numpy as np
from EM_HMM_FR import logGaussian

A = np.array([[0., 1., 0., 0., 0.],
              [0., 0.5, 0.5, 0., 0.],
              [0., 0., 0.5, 0.5, 0.],
              [0., 0., 0., 0.5, 0.5],
              [0., 0., 0., 0., 1.]])

def parse(array,value):
    t=[]
    for item in array:
        t.append(item)
    t.append(value)
    return t

def viterbi_dist_FR(
    mean=np.array([[10., 0., 0.],[5., 2., 9.]]),
    var=np.array([[1.,1.,1.],[1.,1.,1.]]),
    aij=np.array([A, np.dot(A, A), np.dot(np.dot(A, A), A)]),
    obs=np.array([[8., 8., 4., 2., 3., 7.],
                  [0., 2., 2., 10., 5., 9.]])):
    dim,t_len=obs.shape
    NaN=np.full((dim,1),np.nan)

    # initialize
    mean=np.concatenate((NaN,mean),1)
    mean=np.concatenate((mean,NaN),1)

    var=np.concatenate((NaN,var),1)
    var=np.concatenate((var,NaN),1)
    aij[aij.shape[0]-1][aij.shape[1]-1]=1
    timing =np.array([i for i in range(1,t_len+2)])
    m_len=mean.shape[1]
    fjt=np.full((m_len,t_len),-np.inf)
    s_chain=np.empty((m_len,t_len),dtype=object)

    # at t=0
    dt=timing[0]-1
    for j in range(1,m_len-1):
        fjt[j,0]=np.log(aij[0,j])+logGaussian(mean[:,j],var[:,j],obs[:,0])
        if fjt[j,0]>-np.inf:
            s_chain[j,0]=np.array([1,j+1])

    # at t 1~timing.len-2
    for t in range(1,t_len):
        dt=timing[t]-timing[t-1]-1
        for j in range(1,m_len-1):
            f_max=-np.inf
            i_max=-1
            f=-np.inf
            for i in range(1,j+1):
                if fjt[i,t-1]>-np.inf:
                    f=fjt[i,t-1]+np.log(aij[i,j])+logGaussian(mean[:,j],var[:,j],obs[:,t])
                if f > f_max:
                    f_max=f
                    i_max=i
            if i_max !=-1:
                s_chain[j,t]=np.array(parse(s_chain[i_max,t-1],j+1))
                fjt[j,t]=f_max

    # at t=end
    dt=timing[timing.shape[0]-1]-timing[timing.shape[0]-2]-1
    fopt=-np.inf
    iopt=-1
    for i in range(1,m_len-1):
        f=fjt[i,t_len-1]+np.log(aij[i,m_len-1])
        if f > fopt:
            fopt=f
            iopt=i

    # optimal result
    if iopt != -1:
        chain_opt=np.array(parse(s_chain[iopt,t_len-1],m_len))

    return fopt
    
