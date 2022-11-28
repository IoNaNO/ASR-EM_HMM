import numpy as np

def logGaussian(m,v,o):
    dim=np.max(v.shape)
    ret=(-1/2)*(dim*np.log(2*np.pi)+np.sum(np.log(v))+np.sum((o-m)*(o-m)/v))
    return ret

def log_sum_alpha(log_alpha_t,aij_j):
    len_x=log_alpha_t.shape[0]
    y=np.full(len_x,-np.inf)
    ymax=-np.inf
    for i in range(0,len_x):
        y[i]=log_alpha_t[i]+np.log(aij_j[i])
        if y[i]>ymax:
            ymax=y[i]
    if ymax == np.inf:
        logsumalpha=np.inf
    else:
        sum_exp=0
        for i in range(0,len_x):
            if ymax == -np.inf and y[i] == -np.inf:
                sum_exp+=1
            else:
                sum_exp+=np.exp(y[i]-ymax)
        logsumalpha=ymax+np.log(sum_exp)
    return logsumalpha

def log_sum_beta(aij_i,mean,var,obs,beta_t1):
    len_x=mean.shape[1]
    y=np.full(len_x,-np.inf)
    ymax=-np.inf
    for j in range(0,len_x):
        y[j]=np.log(aij_i[j])+logGaussian(mean[:,j],var[:,j],obs)+beta_t1[j]
        if y[j]>ymax:
            ymax=y[j]
    if ymax==np.inf:
        logsumbeta=np.inf
    else:
        sum_exp=0
        for i in range(0,len_x):
            if ymax == -np.inf and y[i] == -np.inf:
                sum_exp+=1
            else:
                sum_exp+=np.exp(y[i]-ymax)
        logsumbeta=ymax+np.log(sum_exp)
    return logsumbeta

def EM_HMM_FR(mean, var, aij, obs):
    dim,T=obs.shape
    NaN=np.full((dim,1),np.nan)
    mean=np.concatenate((NaN,mean),1)
    mean=np.concatenate((mean,NaN),1)

    var=np.concatenate((NaN,var),1)
    var=np.concatenate((var,NaN),1)
    
    aij[aij.shape[0]-1][aij.shape[1]-1]=1
    N=mean.shape[1]
    log_alpha=np.full((N,T+1),-np.inf) # initialization
    log_beta=np.full((N,T+1),-np.inf) # initialization

    # calculate alpha
    for i in range(0,N):
        log_alpha[i,0]=np.log(aij[0,i])+logGaussian(mean[:,i],var[:,i],obs[:,i]) # log(alpha)
    for t in range(1,T): # calculate alpha
        for j in range(1,N-1):
            log_alpha[j,t]=log_sum_alpha(log_alpha[1:N-1,t-1],aij[1:N-1,j])+logGaussian(mean[:,j],var[:,j],obs[:,t])
    log_alpha[N-1,T]=log_sum_alpha(log_alpha[1:N-1,T-1],aij[1:N-1,N-1])

    # calculate beta
    log_beta[:,T-1]=np.log(aij[:,N-1])
    for t in range(T-2,-1,-1):
        for i in range(1,N-1):
            log_beta[i,t]=log_sum_beta(aij[i,1:N-1],mean[:,1:N-1],var[:,1:N-1],obs[:,t+1],log_beta[1:N-1,t+1])
    log_beta[N-1,0]=log_sum_beta(aij[0,1:N-1],mean[:,1:N-1],var[:,1:N-1],obs[:,0],log_beta[1:N-1,0])

    # calculate Xi
    log_Xi=np.full((T,N,N),-np.inf)
    for t in range(0,T-1):
        for j in range(1,N-1):
            for i in range(1,N-1):
                log_Xi[t,i,j]=log_alpha[i,t]+np.log(aij[i,j])+logGaussian(mean[:,j],var[:,j],obs[:,t+1])+log_beta[j,t+1]-log_alpha[N-1,T]
    
    # when t==T-1
    for i in range(0,N):
        log_Xi[T-1,i,N-1]=log_alpha[i,T-1]+np.log(aij[i,N-1])-log_alpha[N-1,T]

    # calculate gamma
    log_gamma=np.full((N,T),-np.inf)
    for t in range(0,T):
        for i in range(1,N-1):
            log_gamma[i,t]=log_alpha[i,t]+log_beta[i,t]-log_alpha[N-1,T]
    gamma=np.exp(log_gamma)

    # calculate sum of mean_numerator, var_numerator, aij_numerator and denominator (single data)
    mean_numerator = np.zeros((dim,N),dtype=np.float64) 
    var_numerator = np.zeros((dim,N),dtype=np.float64)
    denominator = np.zeros((N,1),dtype=np.float64)
    aij_numerator = np.zeros((N,N),dtype=np.float64)
    for j in range(1,N-1):
        for t in range(0,T):
            mean_numerator[:,j]+=np.dot(gamma[j,t],obs[:,t])
            var_numerator[:,j]+=np.dot(gamma[j,t],obs[:,t])*obs[:,t]
            denominator[j]+=gamma[j,t]

    for i in range(1,N-1):
        for j in range(1,N-1):
            for t in range(0,T):
                aij_numerator[i,j]+=np.exp(log_Xi[t,i,j])
    log_likelihood=log_alpha[N-1,T]
    likelihood=np.exp(log_likelihood)

    return mean_numerator,var_numerator,aij_numerator,denominator,log_likelihood,likelihood