
import sys
import numpy as np
import cv2
import os

from numba import jit

from tqdm import tqdm

#The length of an image
sizeIm = 6980



@jit
def proportionWB(im):
	"""
	Take an image in black and white and return the proportion of white pixels.
	It takes pixels in the middle of the image.
	
	Parameters
	----------
	-im: image in black/white to take the proportion.
	
	Return
	------
	
	Number of white pixels over all pixels in an area in the middle of the image.
	"""
	
	#We take the middle of the image to be sure to have the strip
	#and don't have a column of full balck pixels
	imReduce= im[0:50, 2000:5000]
	#Pick up the shape of the image
	rows,cols = imReduce.shape
	#Count the number of black and white pixels
	label, count = np.unique(imReduce, return_counts=True)
	#White pixel are the 255 value, and take the second place in the array
	nW=count[1]
	#Return the number of whithe pixel over the area
	return (float(nW)/(cols*rows))



def preparation(path):
	"""
	From a folder, take all images and align them to make an image of all of them combine.
	The combine images are place in the folder ResCombine
	Parameters
	----------
	-path: path of the folder where to take all images.
	
	"""

	#Pick up all files in the path folder
	filelist= os.listdir(path+'/ResultPreparation/')
	#List to place differents image from the side
	listUp=[]
	listDown=[]
	for fi1 in filelist:
		#Creating the image and convert it in an black/white image	
		im1=cv2.imread(path+'/ResultPreparation/'+fi1)
		print(path+'/ResultPreparation/'+fi1)
		im1=cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
		if 'Up' in fi1:
			listUp.append(fi1)
		if 'Down' in fi1:
			listDown.append(fi1)
	#Creating the combine image for the differents sides	
	imResCombUp=prepDiff(path,listUp,'Up')
	imResCombDown=prepDiff(path,listDown,'Down')
	
def prepDiff (path,listIm,name):
	"""
	Create an image from all te image in the list. All images are align with a maximum stacking, and apply to create an image with differents shades of gray.
	Each shades represent a number of layers of superimposed images.
	
	Parameters
	----------
	-listIm: list of image
	
	-name: for the file at the end of creation, the image will have the following name: 'ResCombine'+name'.tiff'
	
	Return
	------
	
	The superimposed image of all given images.
	
	"""
	
	#Create an full black image
	container = np.full((50,sizeIm),0,np.uint8)
	#Pick up the value of the length of the list before the pop
	lenListIm = len(listIm)
	#The first image of the list will serve as templates for the alignment of other images
	ref= listIm.pop(0)

	#Create the reference
	imRef= cv2.imread(path+'/ResultPreparation/'+ref)
	imRef=cv2.cvtColor(imRef,cv2.COLOR_BGR2GRAY)
	#Each layer will have the following value 
	grayScale=float(1)/lenListIm
	print 'GrayScale: ', float(1)/lenListIm
	#add the first layer to the container, the template
	container = cv2.addWeighted(container,1,imRef, grayScale,0)
	#Do the same for all images
	for fi in listIm:
		imComp=cv2.imread(path+'/ResultPreparation/'+fi)
		imComp=cv2.cvtColor(imComp,cv2.COLOR_BGR2GRAY)
		#Add a layer to the container for each image
		container = differences(path,container,imRef,imComp,grayScale,name)
	return container
	
	
def differences(path,container,imRef,imComp,grayScale,name):
	"""
	With a reference image and an other image, superimposed the second to a container image with the better alignment possible.
	
	Parameters
	----------
	
	-container: image with superimposed images
	-imRef: template image
	-imComp: image to compare to the template
	-grayScale: scale of gray to apply to white pixels when adding imComp to container 
	-name: name of final image
	
	Return
	------
	
	Return the container image
	
	"""
	
	#Create the first superimposed image
	imBlend = cv2.addWeighted(imRef,0.6,imComp,0.6,0)
	#apply a threshold to get where pixels are white in the two images
	ret,thresh = cv2.threshold(imBlend,220,255,cv2.THRESH_BINARY)
	
	maxAlign = 0
	#The proportion of black and white pixels will be used as value to compare a superimposed image from an other
	maxProp = proportionWB(thresh)
	#Compare the two images for the 2940 first pixels, with a offset of 1 after each iteration
	#2940 represent the average value of the length of one turn of a wheel in pixels for Fellowes 90S
	for offset in tqdm(range(1,2940)):
		#Create a container for the image to compare but with an offset
		imCompTemp = np.full((50,sizeIm),0,np.uint8)
		imCompTemp[ 0:50, offset: sizeIm] = imComp[0:50, 0:sizeIm-offset]
		#Superimposed
		imBlend = cv2.addWeighted(imRef,0.6,imCompTemp,0.6,0)
		#threshold
		ret, thresh = cv2.threshold(imBlend,220,255,cv2.THRESH_BINARY)
		#Evaluate
		propTemp= proportionWB(thresh)
		#Change the maximum
		if propTemp>maxProp: maxAlign, maxProp = offset, propTemp
	#Create a container
	imResTemp = np.full((50,sizeIm),0,np.uint8)
	#Create the solution
	imResTemp[0:50, maxAlign:sizeIm] = imComp[0:50, 0:sizeIm-maxAlign]
	container = cv2.addWeighted(container,1,imResTemp, grayScale,0)

	#If a view is needed
	'''
	cv2.namedWindow('container', cv2.WINDOW_NORMAL)
	cv2.imshow('container',container)

	k = cv2.waitKey(0) & 0xFF
	if k==27:
		cv2.destroyAllWindows()
	'''
	#Create the file
	cv2.imwrite(path+'/FFT/ResCombine'+name+'.tiff', container)	
	return container

