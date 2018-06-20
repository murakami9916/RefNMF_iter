
import numpy as np
import matplotlib.pyplot as plt
import sys
import createRef

def NMF(Y, i, R=2, n_iter=10000, init_H=[], init_U=[], verbose=False):
	print("[{}]".format(i))
	eps = np.spacing(1)

	# size of input spectrogram
	M = Y.shape[0]
	#print(M)
	N = Y.shape[1]
	#print(N)
    
	# initialization
	if len(init_U):
		U = init_U
		R = init_U.shape[0]
	else:
		#U = np.random.rand(R,N);
		U = np.ones((R,N))

	if len(init_H):
		H = init_H;
		R = init_H.shape[1]
	else:
		#H = np.random.rand(M,R)
		H = np.loadtxt("Ref_{0:03}.csv".format(i), delimiter = ",")
        
	# array to save the value of the euclid divergence
	cost = np.zeros(n_iter)

	# computation of Lambda (estimate of Y)
	Lambda = np.dot(H, U)

	# iterative computation
	s = np.zeros(n_iter)
	for i in range(n_iter):

		# compute euclid divergence
		cost[i] = euclid_divergence(Y, Lambda)

		# update H
		H *= np.dot(Y, U.T) / (np.dot(np.dot(H, U), U.T) + eps)

		# update U
		U *= np.dot(H.T, Y) / (np.dot(np.dot(H.T, H), U) + eps)

		# recomputation of Lambda
		Lambda = np.dot(H, U)

	return [H, U, cost]

def euclid_divergence(Y, Yh):
    d = 1 / 2 * (Y ** 2 + Yh ** 2 - 2 * Y * Yh).sum()
    return d

if __name__ == '__main__':

	argc = len(sys.argv)
	if(argc<2):
		sys.exit()

	fname = sys.argv[1]
	
	Y = np.loadtxt(fname, delimiter = ",")

	##parameter-------
	start = 1
	stop = 62
	step = 1
	c = [47, 37]
	r = [-20, -20]
	orbit = "2p"
	##-----------------


	iter_nmf = 100
	bestLoss = 1e300
	lossList = np.array([])
	print("start")
	for i in range(iter_nmf):
		##Create Ref--------------------------------------------------------------
		y0 = createRef.runCreate(start, stop, step, c[0], r[0], orbit, i, i)
		y1 = createRef.runCreate(start, stop, step, c[1], r[1], orbit, i, i + iter_nmf)

		y = np.array((y0, y1))

		np.savetxt("Ref_{0:03}.csv".format(i), y.T, fmt = "%.7f", delimiter = ",")
		##------------------------------------------------------------------------

		np.random.seed(i)
		print("[ {} ]".format(np.random.rand()))
		computed_temp = NMF(Y, i, R=2)
		ss = computed_temp[2]
		lossLog = 10 * np.log(ss[-1])
		lossList = np.append(lossList, lossLog)
		print("[ {} ]{}".format(i, lossLog))

		if(bestLoss > lossLog):
			bestLoss = lossLog
			computed = computed_temp

	print('\ndecomposed\n---------------')
	print('H:\n', computed[0])
	print('U:\n', computed[1])
	print('HU:\n', np.dot(computed[0], computed[1]))
	print('cost:\n', computed[2])

	M = Y.shape[0]
	N = Y.shape[1]

	h = computed[0]
	u = computed[1]

	np.savetxt("H.csv", h, fmt="%.5f", delimiter = ",")
	np.savetxt("U.csv", u, fmt = "%.5f", delimiter = ",")
	np.savetxt("R.csv",  computed[2].T, fmt = "%.5f", delimiter = ",")
	

	fig = plt.figure("convergence")
	plt.plot(computed[2].T, marker = "o")
	plt.xlim(0,10)
	
	fig = plt.figure("weightFunc")
	plt.plot(range(N), computed[1].T, marker = "o")

	fig = plt.figure("baseFunc")
	plt.plot(range(M), computed[0])

	fig = plt.figure("LOSS")
	plt.scatter(range(iter_nmf), lossList)

	plt.show()

	mean = np.mean(lossList)
	std = np.std(lossList)
	print("mean:{}".format(mean))
	print("std:{}".format(std))
	np.savetxt("loss.csv",  lossList.T, fmt = "%.5f", delimiter = ",")

