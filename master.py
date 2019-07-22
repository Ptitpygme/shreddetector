import catcher
import exami
import align
import target
import peak

import os
import cv2
import numpy as np

def run(path,fileName, step, visual, thresh1):

	#Step 0
	#Extracting strip from the source image
	#and prepare them to be analyse
	if step <1:
		print 'Catching and exami phase'
		exami.clearResultFolder('/udd/cvolantv/Pictures/ScanDetector/ResultCatcher')
		exami.clearResultFolder('/udd/cvolantv/Pictures/ScanDetector/ResultPreparation')
		pathFile= path+'/'+fileName
		filelist= os.listdir(pathFile)

		for i in filelist:
			catcher.createSet(pathFile,i,1,0)
			exami.preparation('/udd/cvolantv/Pictures/ScanDetector/ResultCatcher','/udd/cvolantv/Pictures/ScanDetector/ResultPreparation',visual,0,thresh1)
		
	#Step 1
	#Create an image of strips align with the best superposition possible
	if step < 2:
		print 'Align phase'
		align.preparation('/udd/cvolantv/Pictures/ScanDetector/ResultPreparation')
	#Step 2
	#Find a small area with clear peaks
	if step < 3:
		print 'target'
		i=60
		l=[]
		while i <= 140 : 
			print 'Target with threshold at ',i,'/255'
			tar=target.target('/udd/cvolantv/Pictures/ScanDetector/FFT/',i,visual)
			l.append(tar[0][1])
			l.append(tar[1][1])
			i+=20
		
		#Recovering the best frequency
	
		compte = dict([(k, l.count(k)) for k in set(l)])
		print compte
		maxFreq= target.maxDict(compte)
		print maxFreq
		
		peakStrip= (1/float(maxFreq)) * 6980
		print 'Peak: ', peak
		filelist =os.listdir('/udd/cvolantv/Pictures/ScanDetector/ResultPreparation')
		areaResult=[]
		peakFloorRoof=[]
		freqListTop=[]
		globalFreq=[]
		for fi in filelist:
			im = cv2.imread(path+'/ResultPreparation/'+fi)
			
			imag = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
			
			floor,roof = peak.heightFinder(imag)
			print floor, roof

			peakFloorRoof.append((floor,roof))
			if visual ==1 :
				im[floor]=(255,100,100)
				im[roof]=(100,255,100)
				cv2.namedWindow('ImFR', cv2.WINDOW_NORMAL)
				cv2.imshow('ImFR',im)
				k = cv2.waitKey(0) & 0xFF
				if k==27:
					cv2.destroyAllWindows()
			
			result=target.acquireTarget(imag,peakStrip,thresh1,visual)
			freqListTop.append(result[1])
			areaResult.append((result,fi))
			globalFreq=globalFreq+result[3]
		third(areaResult)
		
		#meanAmpl=np
		
		topArea1=areaResult[-1]
		topArea2=areaResult[-2]
		topArea3=areaResult[-3]
		
		cpt = dict([(k, freqListTop.count(k)) for k in set(freqListTop)])
		
		
		print 'Top 3 area: ',topArea1,topArea2,topArea3
		print 'Average value of Floor/Roof: ', np.average(np.array(peakFloorRoof),axis=0)
		print 'Total: ', cpt
		peakIdentifier(topArea1)
		peakIdentifier(topArea2)
		peakIdentifier(topArea3)
		
		
		cptGlobalFreq = dict([(k,( globalFreq.count(k),100*globalFreq.count(k)/float(len(globalFreq)))) for k in set(globalFreq)])
		print 'Global Freq: ',cptGlobalFreq 
		
		return topArea1,topArea2,topArea3
		
def guessMeRandom(choice,visual, thresh1):
		
		exami.clearResultFolder('/udd/cvolantv/Pictures/ScanDetector/ResultPreparation')
		

		chosenFile= choice[np.random.randint(len(choice))]
		imageList= os.listdir('/udd/cvolantv/Pictures/ScanDetector/CatchThemAll/'+chosenFile)
		chosenImage= imageList[np.random.randint(len(imageList))]
		exami.shape('/udd/cvolantv/Pictures/ScanDetector/CatchThemAll/'+chosenFile,chosenImage,'/udd/cvolantv/Pictures/ScanDetector/ResultPreparation',0,thresh1)
		
		i=60
		l=[]
		while i <= 140 : 
			print 'Target with threshold at ',i,'/255'
			tar=target.target('/udd/cvolantv/Pictures/ScanDetector/FFT/',i,0)
			l.append(tar[0][1])
			l.append(tar[1][1])
			i+=20
		
		#Recovering the best frequency
	
		compte = dict([(k, l.count(k)) for k in set(l)])
		print compte
		maxFreq= target.maxDict(compte)
		print maxFreq
		
		peakStrip= (1/float(maxFreq)) * 6980
		filelist =os.listdir('/udd/cvolantv/Pictures/ScanDetector/ResultPreparation')
		areaResult=[]
		peakFloorRoof=[]
		freqList=[]
		globalFreq=[]
		for fi in filelist:
			im = cv2.imread('/udd/cvolantv/Pictures/ScanDetector/ResultPreparation/'+fi)
			
			imag = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
			
			floor,roof = peak.heightFinder(imag)
			print floor, roof

			peakFloorRoof.append((floor,roof))
			if visual ==1:
				im[floor]=(255,100,100)
				im[roof]=(100,255,100)
				cv2.namedWindow('ImFR', cv2.WINDOW_NORMAL)
				cv2.imshow('ImFR',im)
				k = cv2.waitKey(0) & 0xFF
				if k==27:
					cv2.destroyAllWindows()
			
			result=target.acquireTarget(imag,peakStrip,thresh1,visual)
			freqList.append(result[1])
			areaResult.append((result,fi))
			globalFreq=globalFreq+result[3]
		
		
		cpt = dict([(k, freqList.count(k)) for k in set(freqList)])
		cptGlobalFreq = dict([(k,( globalFreq.count(k),100*globalFreq.count(k)/float(len(globalFreq)))) for k in set(globalFreq)])
		print '-------------------------------------------------------------------------'
		print 'Average value of Floor/Roof: ', np.average(np.array(peakFloorRoof),axis=0)
		print 'Total: ', cpt
		print 'Global freq of areas', cptGlobalFreq
		print '-------------------------------------------------------------------------'
		print 'Width strip: ',getWidthStripSpecified('/udd/cvolantv/Pictures/ScanDetector/CatchThemAll/'+chosenFile+'/'+chosenImage)
		print '-------------------------------------------------------------------------'
		
		
		
		waiting= ""
		
		while waiting != "OK":
			waiting = raw_input("Write OK when you are ready to see the solution: ")
		print chosenImage
		return areaResult
		
		
		
		
		
		
		
		
def third(sub_li):
	sub_li.sort(key = lambda x: x[0][2])
	return sub_li



def peakIdentifier(masterResult):
	pos = masterResult[0][0]
	freq=masterResult[0][1]
	nameIm= masterResult[1]
	

	
	
	im = cv2.imread('/udd/cvolantv/Pictures/ScanDetector/ResultPreparation/'+nameIm)
	imag = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
	imArea=imag[0:50, pos:pos+350]
	
	floor,roof = peak.heightFinder(imArea)
	
	im[floor]=(255,100,100)
	im[roof]=(100,255,100)
	
	cv2.namedWindow('ImRes', cv2.WINDOW_NORMAL)
	cv2.imshow('ImRes',imArea)
	cv2.namedWindow('ImFR', cv2.WINDOW_NORMAL)
	cv2.imshow('ImFR',im)
	k = cv2.waitKey(0) & 0xFF
	if k==27:
		cv2.destroyAllWindows()
		
		
def getAverageWidthStrip():
	path='/udd/cvolantv/Pictures/ScanDetector/ResultCatcher'
	fileList= os.listdir(path)
	averageList=[]
	for fi in fileList:
		ave=getWidthStripSpecified(path+'/'+fi)
		if ave !=-1:
			averageList.append(ave)
			arr=np.array(averageList)
	return np.average(arr), np.std(arr),np.median(arr), np.percentile(arr, 25), np.percentile(arr,75)
		
def getWidthStripSpecified(im):	
	im=cv2.imread(im)
	im=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
		
	rows,cols = im.shape
	if (cols > 1000):
		im=im[0:rows-1, cols-6990: cols-20]
		ave = np.average(np.sum(im,axis=0)//255) 
		return ave
	else:
		return -1
