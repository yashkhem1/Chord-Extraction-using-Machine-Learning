'''
Mohd Safwan
Arpit Aggrawal
Institute Project
IIT Bombay
'''
import os
import soundfile as sf
import scipy.io.wavfile
import pickle
import numpy as np
from sklearn.kernel_approximation import AdditiveChi2Sampler
from scipy.signal import butter, lfilter
import pydub
import filetype
# This contains various utilities required during our working

N_to_C = {1: 'A', 2: 'Am', 3: 'Bm', 4: 'C', 5: 'D',
          6: 'Dm', 7: 'E', 8: 'Em', 9: 'F', 10: 'G'}
C_to_N = {'A': 1, 'Am': 2, 'Bm': 3, 'C': 4, 'D': 5,
          'Dm': 6, 'E': 7, 'Em': 8, 'F': 9, 'G': 10}

# Converts Stereo Audio to Mono Audio
# Does it by simply averaging down the audio from left and right channels


def convert(myAudioFile, path='./'):
    fmt = filetype.guess(myAudioFile).extension
    sound_stereo = pydub.AudioSegment.from_file(
        os.path.join(path, myAudioFile), format=fmt)
    mono_list = sound_stereo.split_to_mono()
    if len(mono_list) == 1:
        print('File contains Mono channel only. Can\'t enhance')
        sound_stereo.export(myAudioFile.rsplit('.')[0] + '.wav', format='wav')
        return myAudioFile.rsplit('.')[0] + '.wav'
    sound_monoL = mono_list[0]
    sound_monoR = mono_list[1]
    sound_monoR_inv = sound_monoR.invert_phase()
    sound_CentersOut = sound_monoL.overlay(sound_monoR_inv)
    sound_CentersOut.export(myAudioFile.rsplit('.')[0] + '.wav', format='wav')
    return myAudioFile.rsplit('.')[0] + '.wav'


def convStoM(y):
    y = y.astype(float)
    mono_y = y[:, 0] / 2 + y[:, 1] / 2
    return mono_y


def mPCP(y, fs):
    if len(y.shape) == 2:
        y = convStoM(y)
    n = np.size(y)
    k = int(n / 2)
    y = (np.square(abs(np.fft.rfft(y))[:k]))
    pcp = np.zeros(12, dtype=float)
    fref = 130.8
    M = np.zeros(k)
    M[0] = -1
    for l in range(1, k):
        M[l] = round(12 * np.log2((fs / fref) * (l / n))) % 12
    for i in range(0, 12):
        pcp[i] = np.dot(y, (M == (i * np.ones(k))))
    if sum(pcp) == 0:
        return np.zeros(12, dtype=float)
    pcp = pcp / sum(pcp)
    return pcp


"""
ARPIT BHAI CHUTIYA GYA HAI KYA
BHENCHOD YE KYA KYA DAAL DIA
BUTTER SAMPLER LODA LASSAN
BC SIGNAL PROCESSING KA SHAUK HAI
TO CHALE JA BC ELEC CSP ME
"""


def bandpass(l, u, fs, order=5):
    nyq = 0.5 * fs
    low = l / nyq
    high = u / nyq

    b, a = butter(order, [low, high], btype='band')
    return b, a


def bandpass_filter(data, llimit, ulimit, fs, order=5):
    b, a = bandpass(llimit, ulimit, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Returns part of a file from a start time to the time the audio needs to played
# The name of output file is given by name using the extension of output file


# KYA FALTU FUNCTION BANAYA ARPIT NE BC
# APUN NE COMMENT KR DIA


def make_part(file, start, time, name):
    '''
    os.system(cmd)
    return
    '''
    song = pydub.AudioSegment.from_wav(file)
    part = song[int(float(start) * 1000):int(float(start)
                                             * 1000) + int(float(time) * 1000)]
    part.export(name, format='wav')
# Returns all parts of file at a time interval of 0.1 seconds
# Though we have refrained ourselves from using it in chord sequencing
# due to memory issues with this function


def all_part(file):
    # The commented part is really bad code written by Arpit.
    # However I have fixed most of it.
    '''
    f = sf.SoundFile(file)
    duration = len(f)/f.samplerate
    i = 0
    while i + 0.1 < duration :
        output_name = "output" + str(int(int(i*100)/10)) + ".wav"
        make_part(file, str(i), "0.1", output_name)
        i += 0.1
    return
    '''
    song = pydub.AudioSegment.from_wav(file)
    duration = song.duration_seconds * 1000
    i = 0
    while i + 100 < duration:
        output_name = 'output' + str(int(int(i * 100) / 10)) + '.wav'
        song[i:i + 100].export(output_name, format='wav')
        i += 100


# Returns the chord in a file using the specified model
def find_chord(model, file, code):
    fs, y = scipy.io.wavfile.read(file)
    y = bandpass_filter(y, 20, 7000, fs, order=5)
    X = mPCP(y, fs).reshape(1, -1)
    sampler = AdditiveChi2Sampler()
    if sum(X.ravel()) == 0:
        return'__'
    if code == 1:
        X = sampler.fit_transform(X)
    pred = model.predict(X)
    # print(pred)
    return NtoC(pred[0])

# This analyses a file and returns all_chords of the file
# present at some interval i
# This interval is optimized to give best results
# We have chosen 0.2 seconds to be the optimization value


def analyse(file, model, code):
    f = sf.SoundFile(file)
    duration = len(f) / f.samplerate
    i = 0
    all_chords = []
    while i + 0.5 <= duration:
        o_name = "output.wav"
        make_part(file, str(i), "0.5", o_name)
        i += 0.5
        all_chords.append(find_chord(model, o_name, code))
        os.remove("output.wav")
    return all_chords

# It returns the sequence of chord in an audio file using the specified model
# Returns chord at an interval of .6 second so the maximum allowed uncertainity
# considering error to be present only in the sequencer is .6 seconds


def chord_sequence(model, file, code):
    file = convert(file)
    print(file)
    f = sf.SoundFile(file)
    duration = len(f) / f.samplerate
    i = 0
    final_chords = []
    while duration > 0:
        o_name = "foo.wav"

        if duration > 1.5:
            make_part(file, str(i), "1.5", o_name)
        else:
            if duration > 0.5:
                make_part(file, str(i), str(duration), o_name)
            else:
                final_chords.append("__")
                break
        analysis = analyse(o_name, model, code)

        final_chords.append(max(set(analysis), key=analysis.count))
        i += 0.5
        duration -= 0.5
        os.remove("foo.wav")
    return final_chords
# Maps numbers back to chord


def NtoC(n):
    if n in range(1, 11):
        return N_to_C[n]
# Inverse of the above bijection


def CtoN(c):
    return C_to_N[c]
