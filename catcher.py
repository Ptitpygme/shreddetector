import numpy as np
import cv2
import os
from matplotlib import pyplot as plt


#Create a set of image of strips from an image with all the strips
def createSet(path, image,clear, visual):
	#We clearing the folder of results
	if clear == 1:
		clearResultFolder('/udd/cvolantv/Pictures/ScanDetector/ResultCatcher')
	
	filename=path+'/'+image
	print filename
	
	#Attempting to get the contours
	kernel = np.ones((5,5),np.uint8)
	ori = cv2.imread(filename)
	print ori.shape
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
			'''
			if dst != -1:
			'''
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
	if (xW-x) > 6000:	
		capt = dst[y:yH, x:xW]
		cv2.imwrite(path,capt)
		return dst
	else:
		return -1
	
def clearResultFolder(path):
	filelist= os.listdir(path)
	print filelist
	for f in filelist:
		os.remove(path+'/'+f)
		
		
		
		
