import numpy as np
import matplotlib.pyplot as plt
import sys

def pseudoVoigtFunc(x, a, c, w, l):
	gauss = pow(2.0, -pow((x - c) / w, 2))
	lorentz = 1.0 / (1.0 + pow((x - c) / w, 2))
	voigt = a * ( (1-l) * gauss + l * lorentz )
	return voigt

def plot(x, y):
	plt.plot(x,y)

def calRand(s, vmax, vmin):
	np.random.seed(s)
	return np.random.rand() * (vmax - vmin) + vmin

def getFunc(start, stop, step, a, c, w, l, r, orbit, numSeed):
	ratio = {"1s":0, "2p":1/2, "3d":2/3}

	energy = np.arange(start, stop, step)
	intensity_main = np.array([])
	for i in energy:
		intensity_main = np.append( intensity_main, pseudoVoigtFunc( i, a, c, w, l) )

	intensity_sub = np.array([])
	for i in energy:
		intensity_sub = np.append( intensity_sub, pseudoVoigtFunc( i, a*ratio[orbit], c + r, w, l) )

	if(len(intensity_main) == len(intensity_sub)):
		intensity = intensity_main + intensity_sub

	#np.savetxt("Ref_{0:03}.csv".format(numSeed), intensity.T, fmt = "%.7f", delimiter = ",")
	#plot(energy, intensity)

	return intensity

def runCreate(start, stop, step, c, r, orbit, i, s):
	aMax = 1.0
	aMin = 0.1
	wMax = 0.5
	wMin = 0.2

	a_rand = calRand(int(s), aMax, aMin)
	print("a:{}".format(a_rand))
	w_rand = calRand(int(s), wMax, wMin)
	print("w:{}".format(w_rand))
	return getFunc(start, stop, step, a_rand, c, w_rand, 0.15, r, orbit, i)

if __name__ == '__main__':
	##parameter
	start = 160
	stop = 165
	step = 0.1
	c = [162, 162.5]
	r = [1, 1]
	orbit = "2p"

	numIter = 100

	for i in range(numIter):
		y0 = runCreate(start, stop, step, c[0], r[0], orbit, i, i)
		y1 = runCreate(start, stop, step, c[1], r[1], orbit, i, i + numIter)

		if(False):
			##plot--------------------------------
			plot(np.arange(start, stop, step), y0)
			plot(np.arange(start, stop, step), y1)
			plt.show()
			##------------------------------------

		y = np.array((y0, y1))

		np.savetxt("Ref_{0:03}.csv".format(i), y.T, fmt = "%.7f", delimiter = ",")

	