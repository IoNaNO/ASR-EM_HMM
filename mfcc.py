# @berif:   ASR Assignment1
# @author:  1953921 陈元哲
import decimal
import math
import python_speech_features as psf
import matplotlib.pyplot as plt
import numpy as np
import scipy.fftpack
import scipy.io

# Func for Round half up
def round_half_up(number):
    return int(decimal.Decimal(number).quantize(decimal.Decimal('1'), rounding=decimal.ROUND_HALF_UP))

def my_mfcc(filename='H1228G_520.wav',bank_no=26,lifter=22):
    # Get the file
    # filename=input('Please input the name of audio file: ')
    sr,y=scipy.io.wavfile.read(filename,mmap=False)
    # y= y[0:int(3.5 * sr)]

    # Pre-emphasis
    alpha=0.97
    signal=np.append(y[0],y[1:]-alpha*y[:-1])

    # Framing&Windowing
    ## set frame size and frame stride
    f_size=0.025
    f_stride=0.01
    f_len=f_size*sr
    f_len=int(round_half_up(f_len))
    f_step=f_stride*sr
    f_step=int(round_half_up(f_step))
    s_len=len(signal)
    if s_len <= f_len:
        numframes = 1
    else:
        numframes = 1 + int(math.ceil((1.0*s_len - f_len)/f_step))
    padlen=int((numframes-1)*f_step+f_len)
    zeros=np.zeros((padlen-s_len,))
    padsignal=np.concatenate((signal,zeros))

    indices = np.tile(np.arange(0,f_len),(numframes,1)) + np.tile(np.arange(0,numframes*f_step,f_step),(f_len,1)).T
    indices = np.array(indices,dtype=np.int32)
    frames = padsignal[indices]
    frames*=np.tile(np.hanning(f_len),(numframes,1))

    # STFT
    NFFT=512
    mag_frames=np.absolute(np.fft.rfft(frames,NFFT))
    pow_frames=((1.0/NFFT)*((mag_frames)**2))

    # Mel-filter bank
    fb=psf.base.get_filterbanks(nfilt=bank_no,nfft=NFFT,samplerate=sr,lowfreq=0,highfreq=None)
    mel_frames=np.dot(pow_frames,fb.T)
    mel_frames=np.where(mel_frames==0,np.finfo(float).eps,mel_frames)   ## Aviod problem with log

    # Log()
    log_mel_frames=np.log(mel_frames)

    # DCT
    mfcc = scipy.fftpack.dct(log_mel_frames, type=2, n=None, axis=1, norm='ortho', overwrite_x=False)
    n_mfcc=13
    mfcc = mfcc[:,1:(n_mfcc+1)] 
    mfcc = psf.base.lifter(cepstra=mfcc, L=lifter)
    ## Energies
    energy=np.sum(pow_frames,1)
    energy=np.where(energy==0,np.finfo(float).eps,energy)
    mfcc[:,0]=np.log(energy)

    return mfcc
