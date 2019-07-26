import numpy as np
import cv2
import os
from matplotlib import pyplot as plt


#Create a set of image of strips from an image with all the strips
def createSet(path, fileIm, image, visual=0):
	"""
	From an image, with a green background, extract all strips on the image.
	It's possible for the program to take border of the image like a strip.
	
	Parameters
	----------
	path: path to the folder of the image to treat
	fileIm: name of the folder of images
	image: Image with strips of paper to extract
	visual:
		0 - Don't create windows of the captures
		1 - Open windows to show what got capture
		2 - Open windows and draw contours of catching box
		
		

	"""
	
	filename=path+'/'+fileIm+'/'+image
	
	
	#Creating the original image
	ori = cv2.imread(filename)
	#Cuting the borders to avoid some to catch white background.
	#Can let some white background.
	im = ori[100:6900, 100:9900]
	#Change the image from BGR to a black/white image
	imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
	rows,cols = imgray.shape
	
	#Every pixels with a shade upper than 180 becomes pixels with a shade of 255, other turn completly black.
	ret,thresh = cv2.threshold(imgray,180,255,cv2.THRESH_BINARY)
	
	#The brush will help to treat images against some noises
	brush = np.ones((5,5),np.uint8)
	
	#Operation OPEN and CLOSE will erase every little white dots in black surfaces, and black dots on white surfaces.
	thresh0 = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, brush)
	thresh1 = cv2.morphologyEx(thresh0, cv2.MORPH_CLOSE, brush)
	#Pick up contours of strips
	contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	if visual == 2:
		#Drawn a red contour on the image	
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

			#Recuperation of rotable rectangle
			crop = im[box[1][1]:box[3][1], box[1][0]:box[3][0]]
			if visual == 2:
			#draw a blue box 
				im = cv2.drawContours(im,[box],0,(255,0,0),2)
			#Take the coordiantes of the box
			x,y,w,h = cv2.boundingRect(cnt)
			
			#Select the area of the box
			crop= im[y:y+h,x:x+w]
			#Show coordiantes of the center of rectangle. It's only to help to identify the rectangle on the image
			#Create the image, and give it the name with an unique name
			dst=capture(im,rect,path+'/ResultCatcher/'+image+'sR'+str(i)+'.tiff')

		### Different display windows
			
			if ((visual >= 1) & (dst is not None)):
			
				##Multiple windows 
			
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
	"""
	Return the center of a rectangle
	
	Parameter
	---------
	rect: coordinnates of the rectangles in the image
	
	Returns
	-------
	return the X and Y coordinates of the rectangle
	"""
	
	xCen=rect[0][0]+(rect[1][0]/2)
	yCen=rect[0][1]+(rect[1][1]/2)
	return xCen,yCen

def capture (im, rect, path):
	"""
	Create an image of a rectangle with a strip inside from coordinates of a rectangle.
	
	Parameters
	----------
	im: Image with strips
	rect: coordiantes of rectangle
	path: where to create the file and the name to give to it
	
	Returns
	-------
		A clear rectangle with strip inside. The rectangle is put horizontally
		-1 if it's not a rectangle 
	"""
	rows,cols,biin = im.shape
	#Verify the angle of the rectangle
	if(rect[2]>-50 ):
		#Create a matrice for accomplishing the rotation of the rectangle
		M = cv2.getRotationMatrix2D((rect[0][0],rect[0][1]),rect[2],1)
		#Make the roatation
		dst = cv2.warpAffine(im,M,(cols,rows))
		#Calculated the up left and down right corners coordinates
		y=np.int0(rect[0][1] -(rect[1][1]/2))
		yH=np.int0(rect[0][1] + (rect[1][1]/2))
		x=np.int0(rect[0][0] - (rect[1][0]/2))
		xW=np.int0(rect[0][0] + (rect[1][0]/2))
	else:
		#Same thing but with a image rotate in a different way
		M = cv2.getRotationMatrix2D((rect[0][0],rect[0][1]),rect[2]+90,1)
		dst = cv2.warpAffine(im,M,(cols,rows))
		y=np.int0(rect[0][1] -(rect[1][0]/2))
		yH=np.int0(rect[0][1] + (rect[1][0]/2))
		x=np.int0(rect[0][0] - (rect[1][1]/2))
		xW=np.int0(rect[0][0] + (rect[1][1]/2))

	#We create the image and create the file associated
	capt = dst[y:yH, x:xW]
	#If the image is too small, we dont create the file
	rows,cols,junk = capt.shape
	if cols < rows:
		print('Reject: wrong orientation')
		return	None
	if cols < 7000:
		print('Reject: too short for examination.')
		return None

	cv2.imwrite(path,capt)
	return dst
		
		
		
		
