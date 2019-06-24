from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from mpl_toolkits.mplot3d import axes3d, Axes3D
from scipy.interpolate import griddata

import sys
import numpy as np
import cv2
import os
import copy
from tqdm import tqdm
np.set_printoptions(threshold=sys.maxsize)

path='/udd/cvolantv/Pictures/ScanDetector/ResultPreparation'
sizeIm = 6980

def proportionWB(im):
	imReduce= im[0:50, 1500:5500]
	rows,cols = imReduce.shape
	label, count = np.unique(imReduce, return_counts=True)
	nW=count[1]
	return (float(nW)/(cols*rows))*100



def preparation(path):
	filelist= os.listdir(path)
	print filelist
	
	listUp=[]
	listDown=[]
	for fi1 in filelist:
		'''
		print fi1
		im1=cv2.imread(path+'/'+fi1)
		im1=cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
		print str(proportionWB(im1))
		'''
		if 'Up' in fi1:
			listUp.append(fi1)
		if 'Down' in fi1:
			listDown.append(fi1)
	print listUp
	print listDown
	#imResCombUp=prepDiff(listUp,'Up')
	imResCombDown=prepDiff(listDown,'Down')
	'''
	imResCombUp1 = imResCombUp[0:50 , 0:2940]
	imResCombUp2 = imResCombUp[0:50 , 2940:5880]
	imResCombUp3 = imResCombUp[0:50 , 5880:sizeIm]
	cv2.imwrite('/udd/cvolantv/Pictures/ScanDetector/ResultEvaluation/ImResComUp1.tiff', imResCombUp1)
	cv2.imwrite('/udd/cvolantv/Pictures/ScanDetector/ResultEvaluation/ImResComUp2.tiff', imResCombUp2)
	cv2.imwrite('/udd/cvolantv/Pictures/ScanDetector/ResultEvaluation/ImResComUp3.tiff', imResCombUp3)
	cv2.namedWindow('ImResComUp1', cv2.WINDOW_NORMAL)
	cv2.imshow('ImResComUp1',imResCombUp1)
	cv2.namedWindow('ImResComUp2', cv2.WINDOW_NORMAL)
	cv2.imshow('ImResComUp2',imResCombUp2)
	cv2.namedWindow('ImResComUp3', cv2.WINDOW_NORMAL)
	cv2.imshow('ImResComUp3',imResCombUp3)
	
	cv2.namedWindow('ImRes', cv2.WINDOW_NORMAL)
	cv2.imshow('ImRes',imResCombUp)
	k = cv2.waitKey(0) & 0xFF
	if k==27:
		cv2.destroyAllWindows()	
	'''
	
	imResCombDown1 = imResCombDown[0:50 , 0:2940]
	imResCombDown2 = imResCombDown[0:50 , 2940:5880]
	imResCombDown3 = imResCombDown[0:50 , 5880:sizeIm]
	cv2.imwrite('/udd/cvolantv/Pictures/ScanDetector/ResultEvaluation/ImResComDown1.tiff', imResCombDown1)
	cv2.imwrite('/udd/cvolantv/Pictures/ScanDetector/ResultEvaluation/ImResComDown2.tiff', imResCombDown2)
	cv2.imwrite('/udd/cvolantv/Pictures/ScanDetector/ResultEvaluation/ImResComDown3.tiff', imResCombDown3)
	cv2.imwrite('/udd/cvolantv/Pictures/ScanDetector/ResultEvaluation/ImResComDown.tiff', imResCombDown)
	
	'''cv2.namedWindow('ImResComDown1', cv2.WINDOW_NORMAL)
	cv2.imshow('ImResComDown1',imResCombDown1)
	cv2.namedWindow('ImResComDown2', cv2.WINDOW_NORMAL)
	cv2.imshow('ImResComDown2',imResCombDown2)
	cv2.namedWindow('ImResComDown3', cv2.WINDOW_NORMAL)
	cv2.imshow('ImResComDown3',imResCombDown3)
	'''
	cv2.namedWindow('ImRes', cv2.WINDOW_NORMAL)
	cv2.imshow('ImRes',imResCombDown)
	k = cv2.waitKey(0) & 0xFF
	if k==27:
		cv2.destroyAllWindows()	
	
	

def prepDiff (listIm,name):

	imRes = np.full((50,sizeIm),0,np.uint8)
	ref= listIm.pop(0)
	lenListIm = len(listIm)
	imRef= cv2.imread(path+'/'+ref)
	imRef=cv2.cvtColor(imRef,cv2.COLOR_BGR2GRAY)
	imRes = cv2.addWeighted(imRes,1,imRef, ((lenListIm*10)/float(255)),0)

	firstMark=markAlign(imRef)
	for fi in listIm:
		imComp=cv2.imread(path+'/'+fi)
		imComp=cv2.cvtColor(imComp,cv2.COLOR_BGR2GRAY)
		#imRes = differences(imRes,imRef,imComp,lenListIm,name)
		imRes=align(imRes,firstMark,imComp,3)
	return imRes
	
	
	
	
	
def markAlign (im):
	rows, cols = im.shape
	i=rows-1
	pos =-1
	while np.unique(im[i])[0]!=0:
		i-=1
	pos=np.argmin(im[i], axis =0)
	print i
	print 'pos ', pos
	return pos
	
	
def neighborAlign(first,second):
	diff = abs(first-second)
	oldDiff=diff
	signe = 1
	change = 1
	if first > second : first, second, change = second, first, -1
	while (oldDiff >= diff):
		print '--------------------------------------'
		oldDiff = diff
		firstRes, secondRes, signeRes= second, first, signe
		print 'oldDiff', oldDiff
		second, first = first + ( 2940 * signe ), second
		print 'first: ', first
		print 'second: ', second
		print 'signe: ', signe
		signe = signe * -1
		diff=abs(first-second)
		print 'diff', diff
	return oldDiff, firstRes, secondRes, signeRes*change
		
def align(imRes, firstMark, imComp, nbIm):
	rows, cols = imComp.shape
	secondMark= markAlign (imComp)
	#Get the difference between two mark
	#The positions where marks need to be put
	#And the direction needed tp push the strop
	diff, first, second, signe = neighborAlign(firstMark, secondMark)
	imAlign= np.full((50,sizeIm),0,np.uint8)
	if signe == -1:
		print 'MDRdiff avant: ',diff
		imAlign[0:50, 0:cols-diff]=imComp[0:50, 0:cols-diff]
	if signe == 1:
		print 'LOL '
		print 'diff :',diff, ' rows: ',cols
		imAlign[0:50, diff: cols]=imComp[0:50, diff:cols]
	cv2.namedWindow('ImAlign', cv2.WINDOW_NORMAL)
	cv2.imshow('ImAlign',imAlign)

	k = cv2.waitKey(0) & 0xFF
	if k==27:
		cv2.destroyAllWindows()
	return cv2.addWeighted(imRes,1,imAlign, ((10*nbIm)/float(255)),0)
	
	
	'''
def differences(imRes,imRef,imComp,nbIm,name):
	imBlend = cv2.addWeighted(imRef,0.6,imComp,0.6,0)
	ret,thresh = cv2.threshold(imBlend,220,255,cv2.THRESH_BINARY)
	maxAlign = 0
	maxProp = proportionWB(thresh)
	for i in tqdm(range(1,2940)):
		imCompTemp = np.full((50,sizeIm),0,np.uint8)
		imCompTemp[ 0:50, i: sizeIm] = imComp[0:50, 0:sizeIm-i]
		imBlend = cv2.addWeighted(imRef,0.6,imCompTemp,0.6,0)
		ret, thresh = cv2.threshold(imBlend,220,255,cv2.THRESH_BINARY)
		propTemp= proportionWB(thresh)
		if propTemp>maxProp: maxAlign, maxProp = i, propTemp
	
	imResTemp = np.full((50,sizeIm),0,np.uint8)

	imResTemp[0:50, maxAlign:sizeIm] = imComp[0:50, 0:sizeIm-maxAlign]
	
	imRes = cv2.addWeighted(imRes,1,imResTemp, (nbIm/float(255)*3),0)

	

	cv2.namedWindow('ImRes', cv2.WINDOW_NORMAL)
	cv2.imshow('ImRes',imRes)

	k = cv2.waitKey(0) & 0xFF
	if k==27:
		cv2.destroyAllWindows()

	cv2.imwrite('/udd/cvolantv/Pictures/ScanDetector/ResultEvaluation/ResCombine'+name+'.tiff', imRes)	
	return imRes
	'''


def graphPrinter(im):
	imResize = im[0:50 , 2000:5000]

	fig = plt.figure()
	ax = fig.gca(projection='3d')

	x= np.arange(3000)
	y= np.arange(50)
	X,Y = np.meshgrid(x,y)
	print imResize
	surf = ax.plot_surface(X, Y, imResize, cmap=cm.coolwarm,
	   linewidth=0, antialiased=False)

	# Add a color bar which maps values to colors.
	fig.colorbar(surf, shrink=0.5, aspect=5)

	plt.show()
	
'''	
def colorScale(shadeOfGray):
	res =np.array([])
	for i in range(shadeOfGray):
		B=255-im*2*(255/shadeOfGray)
		if(B<0): B=0 
		R=im*2*(255/shadeOfGray)-255
		if(R<0): R=0
		G=255-(B+R)
		res.apprend(B,G,R)
	return res

def funcColorize(im, scale)
	im = scale[i]
	
def colorized(im, shadeOfGray):
	scale = colorScale(shadeOfGray)
	vfun = np.vectorize(funcColor)
	res = vfun(im, scale)
'''	
	
