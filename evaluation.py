import numpy as np
import cv2
import os
import copy
from tqdm import tqdm

path='/udd/cvolantv/Pictures/ScanDetector/ResultPreparation'

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
		else:
			listDown.append(fi1)
		
	prepDiff(listUp,'Up')
	prepDiff(listDown,'Down')
	

def prepDiff (listIm,name):
	imRes = np.full((50,7000),0,np.uint8)
	ref= listIm.pop(0)
	imRef= cv2.imread(path+'/'+ref)
	imRef=cv2.cvtColor(imRef,cv2.COLOR_BGR2GRAY)
	for fi in listIm:
		imComp=cv2.imread(path+'/'+fi)
		imComp=cv2.cvtColor(imComp,cv2.COLOR_BGR2GRAY)
		imRes = differences(imRes,imRef,imComp,len(listIm),name)
	

def differences(imRes,imRef,imComp,nbIm,name):
	cast = np.full((50,7000),0,np.uint8)
	imBlend = cv2.addWeighted(imRef,0.6,imComp,0.6,0)
	ret,thresh = cv2.threshold(imBlend,220,255,cv2.THRESH_BINARY)
	maxAlign = 0
	maxProp = proportionWB(thresh)
	for i in tqdm(range(1,1290)):
		imCompTemp = cast
		imCompTemp[ 0:50, i: 7000] = imComp[0:50, 0:7000-i]
		ret, thresh = cv2.threshold(imCompTemp,220,255,cv2.THRESH_BINARY)
		propTemp= proportionWB(thresh)
		if propTemp>maxProp: maxAlign, maxProp = i, propTemp
	
	
	if maxAlign< (1290/2):
		imCompTemp = cast
		imCompTemp[ 0:50, maxAlign: 7000] = imComp[0:50, 0:7000-maxAlign]
		
	else:
		imCompTemp = cast
		imCompTemp[ 0:50, maxAlign: 7000] = imComp[0:50, maxAlign:7000]
		
	imRes = cv2.addWeighted(imRes,1,imComp, (nbIm/float(255))*2,1)
	print maxAlign
	print maxProp
	'''
	cv2.namedWindow('ImRes', cv2.WINDOW_NORMAL)
	cv2.imshow('ImRes',imRes)

	k = cv2.waitKey(0) & 0xFF
	if k==27:
		cv2.destroyAllWindows()
	'''
	cv2.imwrite('/udd/cvolantv/Pictures/ScanDetector/ResultEvaluation/ResCombine'+name+'.tiff', imRes)	
	return imRes

