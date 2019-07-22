import numpy as np
import sys
import cv2
import os
import matplotlib.pyplot as plt
np.set_printoptions(threshold=sys.maxsize)


def getTabHeight (im,threshold):

	ret,im = cv2.threshold(im,threshold,255,cv2.THRESH_BINARY)
	return ((im!=0).argmax(axis=0))**2, im

def target(path,threshold,visual):
	
	filelist= os.listdir(path)
	tab=[]
	for fi in filelist:
		im = cv2.imread(path+'/'+fi)
		im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
		h,imag=getTabHeight(im,threshold)
		tab.append(np.fft.rfft(h-np.mean(h)))

	tabFFT = np.array(tab)
	t1=tabFFT[1]
	t2=tabFFT[0]
	
		
		

	
	rows,cols=im.shape	

	
	#num number of samples for accuracy
	t = np.linspace(0, 1, num=cols, endpoint = True)
	T = t[1] - t[0]  # sampling interval 
	N = t1.size
	
	tb1 =np.abs(t1)* 1 / N # 1 / N is a normalization factor
	tb2 =np.abs(t2)* 1 / N
	
	if visual == 1:
		
		
		cv2.namedWindow('ImRes', cv2.WINDOW_NORMAL)
		cv2.imshow('ImRes',imag)

		k = cv2.waitKey(0) & 0xFF
		if k==27:
			cv2.destroyAllWindows()
	
	
		# 1/T = frequency
		f = np.linspace(0, 1 / T, N)
		

		plt.ylabel("Amplitude")
		plt.xlabel("Frequency")
		plt.plot(f, tb1)  
		plt.plot(f, tb2)  

		plt.show()
	
	return getFreq(tb1,20), getFreq(tb2,20)
	
	

def targetSizePeak(im,threshold):
	
	return 1/float(getFreq(fourier(im,threshold)[0],20)[1])*im.shape[1]
	
	
def getFreq(fft, th):
	size = fft.size
	if th == 1:
		print np.max(fft), np.argmax(fft)*2
		return np.max(fft), np.argmax(fft)*2
	else:
		reducFFT = fft[size//th:]

		print 'Ampl / Freq of area: ',np.max(reducFFT), (size//th+np.argmax(reducFFT))*2
		return np.max(reducFFT), (size//th+np.argmax(reducFFT))*2
	
	
def acquireTarget(im,sizePeak,threshold,visual):
	scale=700
	print 'scale: ',scale
	i=0
	resArea=i
	imRed= im[0:50, 3000:]
	area=imRed[0:50, i:(i+scale)]

	rows, cols =imRed.shape
	maxAmpl, bestFreq= -1, -1
	
	listFreq=[]
	while i+scale < cols :
		fou, imFou = fourier(area,threshold)
		#Display area

		colsArea=area.shape[1]
		t = np.linspace(0, 1, num=colsArea)
		T = t[1] - t[0]  # sampling interval 
		N = fou.size
		
		fou = np.abs(fou) * 1 / N
		if visual == 1:
			# 1/T = frequency
			f = np.linspace(0, 1 / T, N)
			

			plt.ylabel("Amplitude")
			plt.xlabel("Frequency")
			plt.plot(f, fou)  # 1 / N is a normalization factor
			plt.show()
			
			cv2.namedWindow('Area', cv2.WINDOW_NORMAL)
			cv2.imshow('Area',imFou)

			k = cv2.waitKey(0) & 0xFF
			if k==27:
				cv2.destroyAllWindows()
			
		areaTargetAmpl, areaTargetFreq = getFreq(fou,50)
		
		if areaTargetAmpl > maxAmpl :
			bestFreq = areaTargetFreq
			maxAmpl = areaTargetAmpl
			resArea=i

		i=i+(scale//2)
		area=imRed[0:50, i:i+scale]
		listFreq.append(areaTargetFreq)
	
	
	print resArea, bestFreq, maxAmpl

		
	return resArea, bestFreq, maxAmpl, listFreq
	
	
	
def fourier(im, threshold):
	rows, cols = im.shape
	h, imFou = getTabHeight(im, threshold)	
	t = np.fft.rfft(h-np.mean(h))
	tab = np.abs(t)* 1 / rows
	return tab, imFou
		
		
		
		
	
		
def maxDict(dicti):
	maxi=0
	res=-1
	for i in dicti:
		if dicti[i]>maxi:
			res=i
			maxi=dicti[i]
	return res
		
		
		
		
		
		
		
	
	
	
