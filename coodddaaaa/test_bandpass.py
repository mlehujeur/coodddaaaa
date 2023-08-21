import numpy as np
import matplotlib.pyplot as plt
from coodddaaaa.butter import BandpassFilter

npts = 200
dt = 1e-6
nyquist = 0.5 / dt

bp = BandpassFilter(
    freqmin=0.05 * nyquist, 
    freqmax=0.2 * nyquist, 
    sampling_rate=1./dt, 
    order=4)

data = np.zeros(npts)
data[npts // 2] = 1.0  # Dirac

plt.figure()
data = np.zeros(npts)
data[npts//2] = 1.0
plt.plot(data)
plt.plot(bp(data, zerophase=False), label="zerophase=False")
plt.plot(bp(data, zerophase=True), label="zerophase=True")
plt.gca().legend()
plt.show()
