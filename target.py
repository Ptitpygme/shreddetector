import numpy as np
import sys
import cv2
import os
import matplotlib.pyplot as plt
np.set_printoptions(threshold=sys.maxsize)


def getTabHeight (im,threshLevel):
	"""
	Return a list of the height. The height is considering as the total number of white pixels in a column.
	
	Parameters
	----------
	*im: image of strip's side in black and white
	*threshLevel: threshold to apply
	
	Return
	------
	Return a list of height for each column.
	"""
	ret,im = cv2.threshold(im,threshLevel,255,cv2.THRESH_BINARY)
	return ((im!=0).argmax(axis=0))**2, im

def target(path,threshLevel,visual):
	"""
	From strip's sides, show frequecy associated with Fourier transformation
	Use this function for combined images.
	Parameter
	---------
	*path: path where are combined strip's side images
	*threshLevel: threshold to apply
	*visual: enable or diable windows to opened
	Return
	------
	Return the frequency of bumps of each combined images.
	
	""" 	
	filelist= os.listdir(path)
	tab=[]
	for fi in filelist:
		im = cv2.imread(path+'/'+fi)
		im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
		#Put the height to the square for a better visuallisation
		h,imag=getTabHeight(im,threshLevel)
		#Substract the average value to the tab
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
	
	
	
	
def getFreq(fft, begin):
	"""
	From an array, return the fequency with the best amplitude
	
	Parameters
	----------
	*fft: array, result of the fft
	*begin: hint where to begin the examination. Help to avoid some results of the fft where the best amplitude could be noises.
	
	Returns
	-------
	*The amplitude of the frequency
	*The frequence associated to the amplitude
	"""


	size = fft.size
	#Special case
	if begin == 1:
		print np.max(fft), np.argmax(fft)*2
		return np.max(fft), np.argmax(fft)*2
	else:
		#Reduce the array to avoid the 1/begin first values
		reducFFT = fft[size//begin:]

		print 'Ampl / Freq of area: ',np.max(reducFFT), (size//begin+np.argmax(reducFFT))*2
		#We need to modify the value because the original array has changed
		return np.max(reducFFT), (size//begin+np.argmax(reducFFT))*2
	
	
def acquireTarget(im,threshold,visual):
	"""
	Examine the strip at differents small areas. Return the area with the best frequency, and values about it.
	
	Parameters
	----------
	*im: image of strip's side
	*threshold: treshold to apply
	*visual: enable or disable windows to opened
	
	Returns
	-------
	1-The image of the best area
	2-The frequency of the area
	3-Th amplitude of the frequency
	4-List of the frequencies
	"""

	#Size of areas
	scale=700
	
	i=0
	resArea=i
	#Reduce image, some image could have columns of full black pixels
	imRed= im[0:50, 3000:]
	#Create first area
	area=imRed[0:50, i:(i+scale)]

	rows, cols =imRed.shape
	maxAmpl, bestFreq= -1, -1
	#List all frequency result from Fourier
	listFreq=[]
	while i+scale < cols :
		#Apply Fourier on the area
		fou, imFou = fourier(area,threshold)
		#Display area

		
		t = np.linspace(0, 1, num=scale)
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
			
		#Extract the best frequency
		areaTargetAmpl, areaTargetFreq = getFreq(fou,50)
		#Change the max
		if areaTargetAmpl > maxAmpl :
			bestFreq = areaTargetFreq
			maxAmpl = areaTargetAmpl
			resArea=i
		#Move the area
		#The area move only of the half of is size on the strip at each step
		i=i+(scale//2)
		area=imRed[0:50, i:i+scale]
		
		listFreq.append(areaTargetFreq)
	
	
	print resArea, bestFreq, maxAmpl

		
	return resArea, bestFreq, maxAmpl, listFreq
	
	
	
def fourier(im, threshold):
	"""
	Apply Fourier to an image.
	
	Parameters
	----------
	*im: image to examine
	*threshold: threshold to apply
	
	Returns
	-------
	1-Fourier's results
	2-Image associated to the results
	"""
	rows, cols = im.shape
	#Get the height of strip's columns and put it to the square
	h, imFou = getTabHeight(im, threshold)
	#Equilibrate values
	t = np.fft.rfft(h-np.mean(h))
	tab = np.abs(t)* 1 / rows
	return tab, imFou
		
		
		
		
	
		
def maxDict(dicti):
	"""
	From a dictionnary, return the max value
	"""
	maxi=0
	res=-1
	for i in dicti:
		if dicti[i]>maxi:
			res=i
			maxi=dicti[i]
	return res
		
		
		
		
		
		
		
	
	
	
