import numpy as np
import scipy.io.wavfile
from os import listdir
from os.path import isfile, join
from utilities import convStoM

#This function takes path of the audio file as its argument
#It extracts our features from audio that are PCP vectors
#These give 12 dimensional vectors where each component of vector
#represents the relative energies of 12 tones that are :
#c, c#, d, d#, e, f, f#, g, g#, a, a#, b
#To achieve this we have first used Real fast fourier transform of signal
#Squared it to achieve the power of each frequency
#Finally the PCP is normalized so that sum of all components add to 1
def pcp(path) :
    fs,y = scipy.io.wavfile.read(path)
    if len(y.shape) == 2 :
        y = convStoM(y)
    n = np.size(y)
    k = int(n/2)
    y = (np.square(abs(np.fft.rfft(y))[:k]))
    pcp = np.zeros(12)
    fref = 130.8
    M = np.zeros(k)
    M[0] = -1
    for l in range(1, k) :
        M[l] = round(12*np.log2((fs/fref)*(l/n)))%12
    for i in range(0, 12) :
        pcp[i] = np.dot(y, (M==(i*np.ones(k))))
    if sum(pcp)==0:
        return np.zeros(12, dtype=float64)
    pcp = pcp/sum(pcp)
    return pcp

#Same as pcp but with arguments modified to take signal data and sample frequency as input
def mPCP(y, fs) :
    if len(y.shape) == 2 :
        y = convStoM(y)
    n = np.size(y)
    k = int(n/2)
    y = (np.square(abs(np.fft.rfft(y))[:k]))
    pcp = np.zeros(12, dtype=float64)
    fref = 130.8
    M = np.zeros(k)
    M[0] = -1
    for l in range(1, k) :
        M[l] = round(12*np.log2((fs/fref)*(l/n)))%12
    for i in range(0, 12) :
        pcp[i] = np.dot(y, (M==(i*np.ones(k))))
    print(sum(pcp),sum(pcp*pcp))
    if sum(pcp)==0:
        return np.zeros(12, dtype=float64)
    pcp = pcp/sum(pcp)
    return pcp

#PCP Extractor extracts pcp of all files present in a target directory
#It puts in a matrix with each row containing pcp vector of each file
def PCP_Extractor(tar_dir) :
    #tar_dir = "A:/ML/Chords-and-Beats-Extraction-using-ML-master/Ver1/Training Set/Guitar_Only/test"
    all_files = [f for f in listdir(tar_dir) if isfile(join(tar_dir, f))]
    PCP = np.zeros((len(all_files), 12))
    i = 0
    for file in all_files :
        PCP[i] = pcp(tar_dir + "/" + file)
        i += 1
    return PCP
