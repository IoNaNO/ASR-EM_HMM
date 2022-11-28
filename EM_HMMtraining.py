import numpy as np
import csv
import pickle as pkl
import matplotlib.pyplot as plt
from EM_HMM_FR import EM_HMM_FR

class HMM:
    def __init__(self):
        self.mean=None
        self.var=None
        self.Aij=None

def calculate_inital_EM_HMM_items(hmm:HMM, num_of_state, num_of_model, sum_of_features, sum_of_features_square, num_of_feature):
    for k in range(0,num_of_model):
        for m in range(0,num_of_state):
            hmm.mean[k,:,m]=sum_of_features.T/num_of_feature
            hmm.var[k,:,m]=sum_of_features_square.T/num_of_feature -hmm.mean[k,:,m]*hmm.mean[k,:,m]
        for i in range(1,num_of_state+1):
            hmm.Aij[k,i,i+1]=0.4
            hmm.Aij[k,i,i]=1-hmm.Aij[k,i,i+1]
        hmm.Aij[k,0,1]=1
    return hmm


def EM_initialization_model(hmm:HMM,training_file_list,DIM,num_of_state,num_of_model):
    sum_of_features=np.zeros((DIM,1),dtype=np.float64)
    sum_of_features_square=np.zeros((DIM,1),dtype=np.float64)
    num_of_features=0

    with open(training_file_list,'r') as inf:
        reader=csv.reader(inf)
        for row in reader:
            if len(row) == 0:
                pass
            file=row[1]
            with open(file,'rb') as data:
                index=pkl.load(data)
                for feature in index.values():
                    sum_of_features+=np.sum(feature,1,keepdims=True)
                    sum_of_features_square+=np.sum(feature**2,1,keepdims=True)
                    num_of_features+=feature.shape[1]
    hmm=calculate_inital_EM_HMM_items(hmm,num_of_state,num_of_model,sum_of_features,sum_of_features_square,num_of_features)
    return hmm
    

def EM_HMMtraining(training_file_list='traningfile_list.csv',DIM=39,num_of_model=10,num_of_state=13):
    hmm=HMM()
    hmm.mean=np.zeros((num_of_model,DIM,num_of_state),dtype=np.float64)
    hmm.var=np.zeros((num_of_model,DIM,num_of_state),dtype=np.float64)
    hmm.Aij=np.zeros((num_of_model,num_of_state+2,num_of_state+2),dtype=np.float64)
    # generate inital HMM
    hmm=EM_initialization_model(hmm,training_file_list,DIM,num_of_state,num_of_model)

    num_of_iteration = 20 # it should be bigger than 10
    log_likelihood_iter = np.zeros(num_of_iteration,dtype=np.float64)
    likelihood_iter = np.zeros(num_of_iteration,dtype=np.float64)

    for iter in range(num_of_iteration):
        with open(training_file_list,'r') as inf:
            # reset value of sum_of_features, sum_of_features_square, num_of_feature, num_of_jump
            sum_mean_numerator = np.zeros((num_of_model,DIM,num_of_state), dtype=np.float64)
            sum_var_numerator = np.zeros((num_of_model,DIM,num_of_state), dtype=np.float64)
            sum_aij_numerator = np.zeros((num_of_model,num_of_state,num_of_state,),dtype=np.float64 )
            sum_denominator = np.zeros((num_of_state, num_of_model),dtype=np.float64)
            log_likelihood = 0
            likelihood = 0

            # read file use csv
            reader=csv.reader(inf)
            for row in reader:
                label=int(row[0])
                file=row[1]
                with open(file,'rb') as data:
                    index=pkl.load(data)
                    for feature in index.values():
                        mean_numerator,var_numerator,aij_numerator,denominator,log_likelihood_i,likelihood_i=EM_HMM_FR(hmm.mean[label-1,:,:],hmm.var[label-1,:,:],hmm.Aij[label-1,:,:],feature)

                        sum_mean_numerator[label-1,:,:]+=mean_numerator[:,1:-1]
                        sum_var_numerator[label-1,:,:]+=var_numerator[:,1:-1]
                        sum_aij_numerator[label-1,:,:]+=aij_numerator[1:-1,1:-1]
                        sum_denominator[:,label-1]+=denominator[1:-1].flatten()
                        log_likelihood+=log_likelihood_i
                        likelihood+=likelihood_i
                    
            # calaulate value of means, variances, aij
            for k in range(num_of_model):
                for n in range(num_of_state):
                    hmm.mean[k,:,n]=sum_mean_numerator[k,:,n]/sum_denominator[n,k]
                    hmm.var[k,:,n]=sum_var_numerator[k,:,n]/sum_denominator[n,k]-hmm.mean[k,:,n]*hmm.mean[k,:,n]
            
            for k in range(num_of_model):
                for i in range(1,num_of_state+1):
                    for j in range(1,num_of_state+1):
                        hmm.Aij[k,i,j]=sum_aij_numerator[k,i-1,j-1]/sum_denominator[i-1,k]
                hmm.Aij[k,num_of_state,num_of_state+1]=1-hmm.Aij[k,num_of_state,num_of_state]
                    
            hmm.Aij[k,num_of_state+1,num_of_state+1]=1
            log_likelihood_iter[iter]=log_likelihood
            likelihood_iter[iter]=likelihood

    # draw traning figure
    plt.figure()
    x=[i for i in range(1,num_of_iteration+1)]
    plt.plot(x,log_likelihood_iter)
    plt.xlabel('iterations')
    plt.ylabel('log likelihood')
    plt.title('number of states: '+str(num_of_state))
    plt.show()
    # save the model
    with open('HMM','wb') as of:
        pkl.dump(hmm,of)

    return hmm


