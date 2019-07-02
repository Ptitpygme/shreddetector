import numpy as np
import cv2
import os
from matplotlib import pyplot as plt


#Create a set of image of strips from an image with all the strips
def createSet(path, image, visual):
	#We clearing the folder of results
	clearResultFolder(path)
	
	filename=path+'/'+image

	'''
	filenameWGray="/udd/cvolantv/Pictures/ScanDetector/A3/A3Gray.jpeg"

	# Experimentation of image opening and masks
	lower_white = np.array([200,120,120])
	upper_white = np.array([255,255,255])

	mask = cv2.inRange(im, lower_white, upper_white)

	result= cv2.bitwise_and(im,im,mask= mask)
	cv2.imwrite(filenameWGray,result)

	cv2.namedWindow('A3', cv2.WINDOW_NORMAL)
	cv2.imshow('A3',im)
	cv2.namedWindow('Mask', cv2.WINDOW_NORMAL)
	cv2.imshow('Mask',mask)
	cv2.namedWindow('Result', cv2.WINDOW_NORMAL)
	cv2.imshow('Result',result)
	k = cv2.waitKey(0) & 0xFF
	if k==27:
		cv2.destroyAllWindows()

	#Thresholding of pixels

	resultGray = cv2.imread(filenameWGray, cv2.COLOR_BGR2GRAY)
	
	#First parameter: image
	#Second parameter: Intensity of gray to activate the thresh
	#Third parameter: Intensity of gray at the result
	#Fourth parameter: Type of result
	ret, thresh = cv2.threshold(resultGray,127,255,cv2.THRESH_BINARY)
	'''

	#Attempting to get the contours
	kernel = np.ones((5,5),np.uint8)
	ori = cv2.imread(filename)

	im = ori[100:6900, 100:9900]
	print("Shape: ",im.shape)
	rows,cols,biin = im.shape

	
	imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
	ret,thresh = cv2.threshold(imgray,180,255,cv2.THRESH_BINARY)
	thresh0 = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
	thresh1 = cv2.morphologyEx(thresh0, cv2.MORPH_CLOSE, kernel)
	contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	if visual == 1:
				im = cv2.drawContours(im, contours, -1, (0,0,255), 3)
	for i in range(len(contours)):
		
		cnt = contours[i]
		if(len(cnt)>50):
			M = cv2.moments(cnt)
			
			#Creation of rectangle
			rect = cv2.minAreaRect(cnt)
			#Creation bounding points

			box = cv2.boxPoints(rect)
			box = np.int0(box)

			print("Box: ",box)
			#Recuperation of rotable rectangle
			crop = im[box[1][1]:box[3][1], box[1][0]:box[3][0]]
			if visual == 1:
				im = cv2.drawContours(im,[box],0,(0,255,0),2)


			x,y,w,h = cv2.boundingRect(cnt)
			#im = cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
			crop= im[y:y+h,x:x+w]
			print ("Coordonnee centerRect: ",centerRect(rect))
			
			dst=capture(im,rect,"/udd/cvolantv/Pictures/ScanDetector/ResultCatcher/"+image+"sR"+str(i)+".tiff")

		### Different display windows

			##Multiple windows (sensible to crash)
			'''
			cv2.namedWindow('Threshold', cv2.WINDOW_NORMAL)
			cv2.imshow('Threshold',thresh)
			cv2.namedWindow('ThresholdOpen', cv2.WINDOW_NORMAL)
			cv2.imshow('ThresholdOpen',thresh1)
			cv2.namedWindow('Contour',cv2.WINDOW_NORMAL)
			cv2.imshow('Contour', im)
			cv2.namedWindow('Rotation',cv2.WINDOW_NORMAL)
			cv2.imshow('Rotation', dst)

			cv2.namedWindow('Strip',cv2.WINDOW_NORMAL)
			cv2.imshow('Strip', crop)
			k = cv2.waitKey(0) & 0xFF
			if k==27:
				cv2.destroyAllWindows()
			'''
			## Unique window, a better presentation but less effective and take time to load
			'''
			images=[thresh,im,dst,crop]
			titles=['Threshold','Contour','Rotation','Strip']
			for i in xrange(4):
				plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
				plt.title(titles[i])
				plt.xticks([]),plt.yticks([])
			plt.show()
			'''

def centerRect(rect):
	xCen=rect[0][0]+(rect[1][0]/2)
	yCen=rect[0][1]+(rect[1][1]/2)
	return xCen,yCen

def capture (im, rect, path):
	print ("rect: ",rect)
	rows,cols,biin = im.shape
	
	if(rect[2]>-50 ):
		M = cv2.getRotationMatrix2D((rect[0][0],rect[0][1]),rect[2],1)
		dst = cv2.warpAffine(im,M,(cols,rows))
		y=np.int0(rect[0][1] -(rect[1][1]/2))
		yH=np.int0(rect[0][1] + (rect[1][1]/2))
		x=np.int0(rect[0][0] - (rect[1][0]/2))
		xW=np.int0(rect[0][0] + (rect[1][0]/2))
	else:
		M = cv2.getRotationMatrix2D((rect[0][0],rect[0][1]),rect[2]+90,1)
		dst = cv2.warpAffine(im,M,(cols,rows))
		y=np.int0(rect[0][1] -(rect[1][0]/2))
		yH=np.int0(rect[0][1] + (rect[1][0]/2))
		x=np.int0(rect[0][0] - (rect[1][1]/2))
		xW=np.int0(rect[0][0] + (rect[1][1]/2))
	print ("y: ",y,"yH: ",yH,"x: ", x,"xW: ", xW)	
	capt = dst[y:yH, x:xW]
	cv2.imwrite(path,capt)
	return dst

def clearResultFolder(path):
	filelist= os.listdir('/udd/cvolantv/Pictures/ScanDetector/ResultCatcher')
	print filelist
	for f in filelist:
		os.remove('/udd/cvolantv/Pictures/ScanDetector/ResultCatcher/'+f)
