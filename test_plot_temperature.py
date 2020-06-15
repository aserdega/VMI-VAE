import numpy as np
from matplotlib import pyplot as plt


'''
tau0=1.0 # initial temperature
np_temp=tau0
ANNEAL_RATE=0.000007#0.000015
MIN_TEMP=0.5
step = 1500

iters = 150000
X = np.array(range(0,iters))
Y = np.zeros(iters)

for i in range(iters):
	if i % step == 1:
		np_temp=np.maximum(tau0*np.exp(-ANNEAL_RATE*i),MIN_TEMP)

	Y[i] = np_temp

plt.plot(Y)
plt.show()
'''


Y = np.array(range(10))
X = np.array(range(10))
plt.bar(X,Y,align='center')
plt.show()
print(Y)