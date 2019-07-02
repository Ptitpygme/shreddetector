import numpy as np
import sys
import cv2
import os
import matplotlib.pyplot as plt

def getTabHeight (path,fi,thresh):
	im = cv2.imread(path+'/'+fi)
	im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
	ret,im = cv2.threshold(im,thresh,255,cv2.THRESH_BINARY)
	cv2.namedWindow('Smoother', cv2.WINDOW_NORMAL)
	cv2.imshow('Smoother',im)
	k = cv2.waitKey(0) & 0xFF
	if k==27:
			cv2.destroyAllWindows()
	return (im!=0).argmax(axis=0)

def fft(path,thresh):
	


	'''
	filelist= os.listdir(path)
	print filelist
	for fi in filelist:
		tab=getTabHeight(path,fi)
	print '--------------------------------'
	print tab
	print len(tab)		
	sp = np.fft.fft(tab)
	freq = np.fft.fftfreq(tab.shape[-1])
	freq=freq[range(6980)]
	sp = sp[range(6980)]
	print 'sp ', sp
	print 'freq ',freq
	
	plt.plot(freq, sp.real)
	plt.show()
	'''
	
	
	#Exemple de la page de scipy
	'''
	Fs = 150.0;  # sampling rate
	Ts = 1.0/Fs; # sampling interval
	t = np.arange(0,1,Ts) # time vector

	ff = 5;   # frequency of the signal
	y = np.sin(2*np.pi*ff*t)

	n = len(y) # length of the signal
	k = np.arange(n)
	T = n/Fs
	frq = k/T # two sides frequency range
	frq = frq[range(n/2)] # one side frequency range

	Y = np.fft.fft(y)/n # fft computing and normalization
	Y = Y[range(n/2)]

	fig, ax = plt.subplots(2, 1)
	ax[0].plot(t,y)
	ax[0].set_xlabel('Time')
	ax[0].set_ylabel('Amplitude')
	ax[1].plot(frq,abs(Y),'r') # plotting the spectrum
	ax[1].set_xlabel('Freq (Hz)')
	ax[1].set_ylabel('|Y(freq)|')

	plt.show()
	'''


	filelist= os.listdir(path)
	tab=[]
	for fi in filelist:
		h=getTabHeight(path,fi,thresh)
		tab.append(np.fft.rfft(h-np.mean(h)))

	tabFFT = np.array(tab)

	print 'tabFFT ',tabFFT

	tabMean = np.mean(tabFFT, axis = 0)
	
	print 'tabMean ',tabMean
	print 'shape ', tabMean.shape
	t = np.linspace(0, 1, num=6980, endpoint = False)
	T = t[1] - t[0]  # sampling interval 
	N = tabMean.size

	# 1/T = frequency
	f = np.linspace(0, 1 / T, N)

	plt.ylabel("Amplitude")
	plt.xlabel("Frequency")
	plt.bar(f[:N // 2], np.abs(tabMean)[:N // 2] * 1 / N, width=2)  # 1 / N is a normalization factor
	plt.show()
	


	'''
	Fs = 18.0;  # sampling rate
	Ts = 1.0/Fs; # sampling interval
	t = np.arange(0,1,Ts) # time vector
	
	filelist= os.listdir(path)
	tab=[]
	for fi in filelist:
		tab.append(np.fft.fft(getTabHeight(path,fi)))
	tabFFT = np.array(tab)
	print 'tabFFT ',tabFFT
	tabMean = np.mean(tabFFT, axis =01)
	
	n=len(tabMean)
	print 'Len tabMean: ', tabMean
	k=np.arange(n)
	T=n/Fs
	frq = k/T # two sides frequency range
	frq = frq[range(n/2)] # one side frequency range

	Y = np.fft.fft(tabMean)/n # fft computing and normalization
	Y = Y[range(n/2)]

	fig, ax = plt.subplots(2, 1)
	ax[0].plot(t,tabMean)
	ax[0].set_xlabel('Time')
	ax[0].set_ylabel('Amplitude')
	ax[1].plot(frq,abs(Y),'r') # plotting the spectrum
	ax[1].set_xlabel('Freq (Hz)')
	ax[1].set_ylabel('|Y(freq)|')

	plt.show()
	'''
