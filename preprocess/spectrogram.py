import numpy as np
from scipy.io import wavfile as wav
from scipy.misc import imsave
from scipy import signal
import matplotlib.pyplot as plt
import sys

rate,data = wav.read(sys.argv[1])
f, t, Sxx = signal.spectrogram(data,rate)
db = np.log10(Sxx)
imsave(sys.argv[1].replace('.wav','.png'),np.flip(db,0))

# plt.pcolormesh(t, f, db)
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.show()
