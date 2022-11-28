import numpy as np
import os
import pickle as pkl
import python_speech_features
import scipy.io
from mfcc import my_mfcc

def norm(data):
    mean=np.mean(data,axis=1)
    std=np.std(data,axis=1)
    transpose =data.T
    for i in range(len(transpose)):
        transpose[i]=(transpose[i]-mean)/std
    return transpose.T

def generate_mfcc_samples(
        indir='./wav',
        in_filter='\.[Ww][Aa][Vv]',
        outdir='./mfcc',
        out_ext='.mfc',
        outfile_format='htk',## htk format
        frame_size_sec = 0.025,
        frame_shift_sec= 0.010,
        use_hamming=1,
        pre_emp=0,
        bank_no=26,
        cep_order=12,
        lifter=22):


    if os.path.exists(outdir):
        pass
    else:
        os.makedirs(outdir)

    # Read wavfiles from ./wav
    infiles={}
    for path,name,files in os.walk(indir):
        if len(name) != 0:
            pass
        infiles[path]=files
    # Extrac mfcc feature
    for key, value in infiles.items():
        if len(value) == 0:
            continue
        if os.path.exists(outdir+'/'+key[-2:]):
            pass
        else:
            os.makedirs(outdir+'/'+key[-2:])
        for file in value:
            filepath=indir+'/'+key[-2:]+'/'+file
            label=file[0]
            if label == 'O':
                label=10
            elif label == 'Z':
                label=11
            else:
                label=int(label)
            # sr,y=scipy.io.wavfile.read(filepath,mmap=False)
            # feature=python_speech_features.mfcc(signal=y,samplerate=sr,winfunc=np.hanning,ceplifter=lifter,nfilt=bank_no,preemph=pre_emp)
            feature=my_mfcc(filepath,bank_no,lifter)
            delta=python_speech_features.base.delta(feature,1)
            deltadelta=python_speech_features.base.delta(feature,2)
            feature_seq=np.hstack((feature,delta,deltadelta)).T

            t={label:feature_seq}
            # Write into .mfc file
            outfile=outdir+'/'+key[-2:]+'/'+file.split('.')[0]+out_ext
            with open(outfile,'wb') as ofile:
                pkl.dump(t,ofile)

    print("Generate mfcc samples Done.\n")
