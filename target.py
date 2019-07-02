import numpy as np
import sys
import cv2
import os
import matplotlib.pyplot as plt
np.set_printoptions(threshold=sys.maxsize)


def getTabHeight (im,threshold):
	
	'''
	cv2.namedWindow('ImRes', cv2.WINDOW_NORMAL)
	cv2.imshow('ImRes',im)

	k = cv2.waitKey(0) & 0xFF
	if k==27:
		cv2.destroyAllWindows()
	'''
	ret,im = cv2.threshold(im,threshold,255,cv2.THRESH_BINARY)
	
	return ((im!=0).argmax(axis=0))**2, im

def target(path,threshold):
	
	filelist= os.listdir(path)
	tab=[]
	for fi in filelist:
		im = cv2.imread(path+'/'+fi)
		im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
		h=getTabHeight(im,threshold)
		tab.append(np.fft.rfft(h-np.mean(h)))

	tabFFT = np.array(tab)
	print tabFFT
	t1=tabFFT[0]
	t2=tabFFT[1]
	
		
	print '-----------------------------------------------------------------'
	print t1
	

	
	rows,cols=im.shape	

	
	#num number of samples for accuracy
	t = np.linspace(0, 1, num=cols, endpoint = False)
	T = t[1] - t[0]  # sampling interval 
	N = cols

	# 1/T = frequency
	f = np.linspace(0, 1 / T, N)
	
	tb1 =np.abs(t1)* 1 / N
	tb2 =np.abs(t2)* 1 / N
	print tb1
	plt.ylabel("Amplitude")
	plt.xlabel("Frequency")
	plt.bar(f[:N // 2], np.abs(tb1)[:N // 2], width=2)  # 1 / N is a normalization factor
	plt.bar(f[:N // 2], np.abs(tb2)[:N // 2], width=2)  # 1 / N is a normalization factor

	print getFreq(tb1)
	print getFreq(tb2)
	plt.show()
	
	
	

def targetSizePeak(im,threshold):
	
	return 1/float(getFreq(fourier(im,threshold)[0])[1])*im.shape[1]
	
	
def getFreq(fft):
	size = fft.size
	reducFFT= fft[size//15 : 14*(size//15	)]
	return np.max(reducFFT), (size//15+np.argmax(reducFFT))*2
	
	
	
def acquireTarget(im,sizePeak,threshold):
	scale=int(20*sizePeak)
	i=1500
	resArea=i
	area=im[0:50, i:(i+scale)]
	im= im [0:50, 3000:]
	rows, cols =im.shape
	maxAmpl, bestFreq= -1, -1
	
	dodgeFirst=1
	
	while i+scale < cols :
		fou, imFou = fourier(area,threshold)
		
		#Display area
		
		colsArea=area.shape[1]
		
		t = np.linspace(0, 1, num=colsArea, endpoint = False)
		T = t[1] - t[0]  # sampling interval 
		N = colsArea
		
		# 1/T = frequency
		f = np.linspace(0, 1 / T, N)
		
		
		plt.ylabel("Amplitude")
		plt.xlabel("Frequency")
		plt.bar(f[:N // 2], np.abs(fou)[:N // 2], width=2)  # 1 / N is a normalization factor
		plt.show()
		
		cv2.namedWindow('Area', cv2.WINDOW_NORMAL)
		cv2.imshow('Area',imFou)

		k = cv2.waitKey(0) & 0xFF
		if k==27:
			cv2.destroyAllWindows()
		
		areaTargetAmpl, areaTargetFreq = getFreq(fou)
		
		if areaTargetAmpl > maxAmpl :
			bestFreq = areaTargetFreq
			maxAmpl = areaTargetAmpl
			resArea=i

		i=i+scale
		area=im[0:50, i:i+scale]
	
	
	cv2.namedWindow('ImRes', cv2.WINDOW_NORMAL)
	cv2.imshow('ImRes',imFou)

	k = cv2.waitKey(0) & 0xFF
	if k==27:
		cv2.destroyAllWindows()
		
	return resArea, bestFreq, maxAmpl
	
	
	
def fourier(im, threshold):
	rows, cols = im.shape
	h, imFou = getTabHeight(im, threshold)	
	t = np.fft.rfft(h-np.mean(h))
	tab = np.abs(t)* 1 / rows
	return tab, imFou
		
		
		
		
		
def targettingArea(path,threshold):
	
	filelist= os.listdir(path)
	tab=[]
	for fi in filelist:
		im = cv2.imread(path+'/'+fi)
		im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
		sizePeak = targetSizePeak(im,threshold)
		print 'sizePeak: ', sizePeak
		print acquireTarget(im,sizePeak,threshold)
	
		
		
		
		
		
		
		
		
		
		
	'''
	print 'tabFFT ',tabFFT

	tabMean = np.mean(tabFFT, axis = 0)
	
	print 'tabMean ',tabMean
	print 'shape ', tabMean.shape
	t = np.linspace(0, 1, num=6980, endpoint = False)
	T = t[1] - t[0]  # sampling interval 
	N = tabFFT.size

	# 1/T = frequency
	f = np.linspace(0, 1 / T, N)

	plt.ylabel("Amplitude")
	plt.xlabel("Frequency")
	plt.bar(f[:N // 2], np.abs(tabFFT)[:N // 2] * 1 / N, width=2)  # 1 / N is a normalization factor
	plt.show()
	'''
	
	
	
