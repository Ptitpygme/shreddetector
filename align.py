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

path='/udd/cvolantv/Pictures/ScanDetector/ResultPreparation'
sizeIm = 6980

def proportionWB(im):
	imReduce= im[0:50, 2000:5000]
	rows,cols = imReduce.shape
	label, count = np.unique(imReduce, return_counts=True)
	nW=count[1]
	return (float(nW)/(cols*rows))
	
def preparation(path):
	filelist= os.listdir(path)
	print filelist
	
	listUp=[]
	listDown=[]
	for fi1 in filelist:
	
		print fi1
		im1=cv2.imread(path+'/'+fi1)
		im1=cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
		print str(proportionWB(im1))
		if 'Up' in fi1:
			listUp.append(fi1)
		if 'Down' in fi1:
			listDown.append(fi1)
	print listUp
	print listDown
	
	imResCombUp=prepDiff(listUp,'Up')
	imResCombDown=prepDiff(listDown,'Down')
	
def prepDiff (listIm,name):

	imRes = np.full((50,sizeIm),0,np.uint8)
	ref= listIm.pop(0)
	lenListIm = len(listIm)
	imRef= cv2.imread(path+'/'+ref)
	imRef=cv2.cvtColor(imRef,cv2.COLOR_BGR2GRAY)
	print 'GrayScale: ', (float(255)/lenListIm)/255
	imRes = cv2.addWeighted(imRes,1,imRef, (float(255)/(lenListIm))/255,0)
	for fi in listIm:
		imComp=cv2.imread(path+'/'+fi)
		imComp=cv2.cvtColor(imComp,cv2.COLOR_BGR2GRAY)
		imRes = differences(imRes,imRef,imComp,lenListIm,name)
	return imRes
	
	
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
	imRes = cv2.addWeighted(imRes,1,imResTemp, (float(255)/nbIm)/255,0)

	
	'''
	cv2.namedWindow('ImRes', cv2.WINDOW_NORMAL)
	cv2.imshow('ImRes',imRes)

	k = cv2.waitKey(0) & 0xFF
	if k==27:
		cv2.destroyAllWindows()
	'''
	cv2.imwrite('/udd/cvolantv/Pictures/ScanDetector/ResultEvaluation/ResCombine'+name+'.tiff', imRes)	
	return imRes

