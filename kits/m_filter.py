import random
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from math import cos,sin

with open(r'D:\aa_learning\pypro\python3-practice\ECG_classification\ECG_kit\ECG_DATA\201801051344_03759378_金祖行_wave.txt','rt') as f:
    x=f.readlines()
x=x[7].strip('\n').split(',')[:200]
x=np.array(x,dtype=np.int32)
fig=plt.figure(1)

# x=np.arange(0,100,0.1,dtype=np.float32)
# x=list(map(sin,x))
# x=x+np.random.randn(len(x))/2
# random.shuffle(x)
# print(type(x[0]))
# print(x)
ax1=fig.add_subplot(2,1,1)
plt.plot(x)
y=signal.medfilt(x,3)
print(x[:10])

N = 10
Fc = 40
Fs = 1600
#numtaps:窗口法
# h =signal.firwin(numtaps=N, cutoff=40, nyq=Fs/2)
# y =signal.lfilter(h, 1.0, x)
ax2=fig.add_subplot(2,1,2)
print(y[:10])
plt.plot(y)
plt.show()
