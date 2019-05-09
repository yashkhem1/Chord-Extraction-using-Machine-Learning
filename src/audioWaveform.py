import matplotlib.pyplot as plot
import numpy as np
import soundfile as sf
#Enter filename below
filename = 't.wav'
y, fs = sf.read(filename)

time=np.linspace(0, len(y)/fs, num=len(y))

plot.figure(1)
plot.plot(time, y)
plot.show()
