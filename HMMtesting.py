import numpy as np
import csv
import pickle as pkl
from EM_HMMtraining import HMM
from viterbi_dist_FR import viterbi_dist_FR
def HMMtesting(hmm:HMM,testing_file_list):
    num_of_model=11
    num_of_error=0
    num_of_testing=0

    with open(testing_file_list,'r') as inf:
        reader=csv.reader(inf)
        for row in reader:
            num_of_testing+=1
            label=row[0]
            label=int(label)
            file=row[1]
            with open(file,'rb') as data:
                index=pkl.load(data)
                for feature in index.values():
                    # predict which the digit is
                    fopt_max=-np.inf
                    digit=-1
                    for p in range(1,num_of_model+1):
                        fopt=viterbi_dist_FR(hmm.mean[p-1,:,:],hmm.var[p-1,:,:],hmm.Aij[p-1,:,:],feature)
                        if fopt > fopt_max:
                            digit=p
                            fopt_max=fopt
                if digit != label:
                    num_of_error+=1
    return (num_of_testing-num_of_error)*100/num_of_testing