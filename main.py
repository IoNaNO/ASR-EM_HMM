import numpy as np
from generate_mfcc_samples import generate_mfcc_samples
from generate_testing_list import generate_tesing_list
from generate_training_list import generate_traning_list
from EM_HMMtraining import EM_HMMtraining
from HMMtesting import HMMtesting

generate_mfcc_samples()
generate_tesing_list()
generate_traning_list()

training_file_list_name = 'trainingfile_list.csv'
testing_file_list_name = 'testingfile_list.csv'

DIM=39 # dimension of a feature vector
num_of_model=11 # number of models: digit '0', digit '1', ... digit '9', digit 'zero'

# the number of states variates from 12 to 15 and tested respectively
num_of_state_start=12
num_of_state_end=15

accuracy_rate=np.zeros(num_of_state_end)

for num_of_state in range(num_of_state_start,num_of_state_end+1):
    hmm=EM_HMMtraining(training_file_list_name,DIM,num_of_model,num_of_state)
    accuracy_rate[num_of_state-1]=HMMtesting(hmm,testing_file_list_name)
    print("state:%d, accuracy rate:%f" % (num_of_state, accuracy_rate[num_of_state - 1]))