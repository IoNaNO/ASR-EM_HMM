import pickle as pkl
from HMMtesting import HMMtesting
from EM_HMMtraining import HMM

testing_file_list_name = 'testingfile_list.csv'
hmm=HMM()
with open('HMM','rb') as model:
    hmm=pkl.load(model)
    acc=HMMtesting(hmm,testing_file_list_name)
    print("state:%d, accuracy rate:%f" % (12, acc))