import catcher
import preparation
import align
import target
import peak

import os
import cv2
import numpy as np

def run(path,fileName, step, threshLevel=170,visual=0):
	""""
	Use it for extracting informations about strips from a same shredder.
	During the process it's possible to get the following informations:
	-The width of the strip
	-Information about teeth of the shredder
	-Results of fast Fourier transformation (fft)
	
	Parameters
	----------
	*path: path where are all needed folders
	*fileName: path or name of the folder where to pick full images
	*step: 
		0-Catch and preparation phase. Extract and creta all images of strips
		1-Align phase. Don't create bases images
		2-Information phase. Don't create any file during this phase.
	*threshLevel: Threshold to apply to images 
	*visual: Enable or not to open windows of results
	
	"""

	#Step 0
	#Extracting strip from the source image
	#and prepare them to be analyse
	if step <1:
		print 'Catching and preparation phase'
		#Clearing result's folder
		preparation.clearResultFolder(path+'/ResultPreparation')
		
		pathFile= path+'/'+fileName
		filelist= os.listdir(pathFile)
		#For each images extract strip and preapare them
		for i in filelist:
			preparation.clearResultFolder(path+'/ResultCatcher')
			catcher.createSet(path,fileName,i)
			preparation.preparation(path+'/ResultCatcher',path+'/ResultPreparation',visual,0,threshLevel)
		
	#Step 1
	#Create an image of strips align with the best superposition possible
	if step < 2:
		print 'Align phase'
		align.preparation(path)
	#Step 2
	#Find a small area with clear peaks
	if step < 3:
		print 'Information phase'
		i=60
		l=[]
		#Try different level of threshold for the align images
		while i <= 140 : 
			print 'Target with threshold at ',i,'/255'
			tar=target.target(path+'/FFT/',i,visual)
			l.append(tar[0][1])
			l.append(tar[1][1])
			i+=20
		
		#Pick up the best frequency
		#Count differents frequencies from different levels
		compte = dict([(k, l.count(k)) for k in set(l)])
		print 'Compte: ',compte
		#Show the most commom frequency
		maxFreq= target.maxDict(compte)
		print 'maxFreq: ',maxFreq
		
		filelist =os.listdir(path+'/ResultPreparation')
		#List where a file and his results are associated
		areaResult=[]
		#List for the base and the top of the bumps
		peakFloorRoof=[]
		#List of most clearer frequencies of each strips
		freqListTop=[]
		#List of all frequencies found
		globalFreq=[]
		for fi in filelist:
		
			
			im = cv2.imread(path+'/ResultPreparation/'+fi)
			
			imag = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
			#pick up the height of bumps
			floor,roof = peak.heightFinder(imag)
			print 'Base and top of bumps: ', floor, roof
			peakFloorRoof.append((floor,roof))
			
			if visual ==1 :
				#Draw to line at the base and top of bumps
				im[floor]=(255,100,100)
				im[roof]=(100,255,100)
				cv2.namedWindow('ImFR', cv2.WINDOW_NORMAL)
				cv2.imshow('ImFR',im)
				k = cv2.waitKey(0) & 0xFF
				if k==27:
					cv2.destroyAllWindows()
			#Extract informations from strips
			result=target.acquireTarget(imag,threshLevel,visual)
			#Add the informations to the differents list
			freqListTop.append(result[1])
			areaResult.append((result,fi))
			globalFreq=globalFreq+result[3]
		#Sort the list
		third(areaResult)
		
		#Get the top 3		
		topArea1=areaResult[-1]
		topArea2=areaResult[-2]
		topArea3=areaResult[-3]
		
		#Count differents frequencies
		cpt = dict([(k, freqListTop.count(k)) for k in set(freqListTop)])
		
		
		print 'Top 3 area: ',topArea1,topArea2,topArea3
		print 'Average value of Floor/Roof: ', np.average(np.array(peakFloorRoof),axis=0)
		print 'Total: ', cpt
		peakIdentifier(path,topArea1)
		peakIdentifier(path,topArea2)
		peakIdentifier(path,topArea3)
		
		
		cptGlobalFreq = dict([(k,( globalFreq.count(k),
								100*globalFreq.count(k)/float(len(globalFreq)))) for k in set(globalFreq)
							])
		print 'Global Freq: ',cptGlobalFreq 
		
		
		
def guessMeRandom(path,choice,threshLevel,visual =0):
		"""
		Pick up randomly a strip in files, extract and show information about it. After an input from the user, the origin of the strip is show.
		
		Warning: use ResultPreparation folder, and we erase the content of them at the beginning
		
		Parameters
		----------
		*path:Path where all folders are
		*choice:List of name's folder of strips
		*treshLevel: threshold to apply
		*visual:enable to open windows of results or not
		
		"""
		
		#Clear 
		preparation.clearResultFolder(path+'/ResultPreparation')
		
		#Random choose
		chosenFile= choice[np.random.randint(len(choice))]
		imageList= os.listdir(path+'/CatchThemAll/'+chosenFile)
		chosenImage= imageList[np.random.randint(len(imageList))]
		preparation.shape(path+'/CatchThemAll/'+chosenFile,chosenImage,path+'/ResultPreparation',0,threshLevel)
		
		#Show global result at differents threshold
		i=60
		l=[]
		while i <= 140 : 
			print 'Target with threshold at ',i,'/255'
			tar=target.target(path+'/FFT/',i,0)
			l.append(tar[0][1])
			l.append(tar[1][1])
			i+=20
		
		#Pick up the best frequency
		#Count differents frequencies from different levels
		compte = dict([(k, l.count(k)) for k in set(l)])
		print 'Compte: ',compte
		#Show the most commom frequency
		maxFreq= target.maxDict(compte)
		print 'maxFreq: ',maxFreq
		

		filelist =os.listdir(path+'/ResultPreparation')
		#List where a file and his results are associated
		areaResult=[]
		#List for the base and the top of the bumps
		peakFloorRoof=[]
		#List of most clearer frequencies of each strips
		freqListTop=[]
		#List of all frequencies found
		globalFreq=[]
		for fi in filelist:
			im = cv2.imread(path+'/ResultPreparation/'+fi)
			
			imag = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
			#pick up the height of bumps
			floor,roof = peak.heightFinder(imag)
			print 'Base and top of bumps: ', floor, roof
			peakFloorRoof.append((floor,roof))

			if visual ==1:
				#Draw to line at the base and top of bumps
				im[floor]=(255,100,100)
				im[roof]=(100,255,100)
				cv2.namedWindow('ImFR', cv2.WINDOW_NORMAL)
				cv2.imshow('ImFR',im)
				k = cv2.waitKey(0) & 0xFF
				if k==27:
					cv2.destroyAllWindows()
			
			#Extract informations from strips
			result=target.acquireTarget(imag,threshLevel,visual)
			#Add the informations to the differents list
			freqList.append(result[1])
			areaResult.append((result,fi))
			globalFreq=globalFreq+result[3]
		
		
		cpt = dict([(k, freqList.count(k)) for k in set(freqList)])
		cptGlobalFreq = dict([(k,( globalFreq.count(k),100*globalFreq.count(k)/float(len(globalFreq)))) for k in set(globalFreq)])
		
		#Display information
		print '---------------------------- Heigth of shredder\'s theeth ----------------------------'
		print 'Average value of Floor/Roof: ', np.average(np.array(peakFloorRoof),axis=0)
		print '------------------------------ Shredder\'s frequencies -------------------------------'
		print 'Total: ', cpt
		print 'Global freq of areas', cptGlobalFreq
		print '------------------------------- Strip\'s informations --------------------------------'
		print 'Width strip: ',getWidthStripSpecified(path+'/CatchThemAll/'+chosenFile+'/'+chosenImage)
		print '--------------------------------------------------------------------------------------'
		
		
		#Wait an correct input before giving the answer
		waiting= ""
		while waiting != "OK":
			waiting = raw_input("Write OK when you are ready to see the solution: ")
		print 'Answer: ',chosenImage
		
		
		
		
def third(li):
	"""
	Sort a list of tuple with the third element.
	"""
	li.sort(key = lambda x: x[0][2])
	return li



def peakIdentifier(path,masterResult):
	"""
	Show images off flatten strip with line representing base and top of bumps
	
	Parameters
	----------
	*path: path where all folders are
	*masterResult: tuple with informations about the strip
	"""
	pos = masterResult[0][0]
	nameIm= masterResult[1]
	
	im = cv2.imread(path+'/ResultPreparation/'+nameIm)
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
		
		
def getStatsWidthStrip(path):
	"""
	Return statistics about the width of strips
	
	Parameter
	---------
	*path: path where all folders are
	
	Return
	------
	return a tuple wth the followed elements:
	-Average
	-Median
	-First percentile
	-Third percentille
	
	"""

	path=path+'/ResultCatcher'
	fileList= os.listdir(path)
	averageList=[]
	for fi in fileList:
		ave=getWidthStripSpecified(path+'/'+fi)
		if ave !=-1:
			averageList.append(ave)
			arr=np.array(averageList)
	return np.average(arr), np.std(arr),np.median(arr), np.percentile(arr, 25), np.percentile(arr,75)
		
def getWidthStripSpecified(im):
	"""
	Return average value of the width of a strip
	
	Parameter
	---------
	*im: image of a strip
	
	Return
	------
	Average with of the strip
	Return -1 if the strip is too short ( not a strip)
	
	"""	
	im=cv2.imread(im)
	im=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
		
	rows,cols = im.shape
	if (cols > 1000):
		im=im[0:rows-1, cols-6990: cols-20]
		ave = np.average(np.sum(im,axis=0)//255) 
		return ave
	else:
		return -1
