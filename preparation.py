import numpy as np
import cv2
import os


def preparation(path, pathRes,visual, clear,threshLevel):
	"""
	From a file of raw strips images, create for each image two news images of each sides of the strip.
	One of the two represent the down side of the strip, the other is the up side.
	Every images are flatten and black and white.
	
	Parameters
	----------
	
	*path: path for the file where to take file to treat
	*pathRes: path fot the file where to put results
	*visual: 
		if 0
			Don't open windows (better for automation)
		if 1
			Show windows of results 
	*clear: 
		if 0
			Don't clear the pathRes file
		if 1
			Clear the pathRes file
	*threshLevel: The value of threshold 
	
	"""
	
	#Clearing file
	if clear == 1:
		clearResultFolder(pathRes)
	
	
	filelist= os.listdir(path)
	for fi in filelist:
		shape(path, fi, pathRes,visual,threshLevel)

#Create the images of differents sides of strip without drops
def shape(path, fi, pathRes,visual,threshLevel):
	"""
	Create the images of differents sides of strip flatten
	
	Parameters
	----------
	
	*path: path for the file where to take file to treat
	*fi: image to treat
	*pathRes: path fot the file where to put results
	*visual: 
		if 0
			Don't open windows (better for automation)
		if 1
			Show windows of results 
	*clear: 
		if 0
			Don't clear the pathRes file
		if 1
			Clear the pathRes file
	*threshLevel: The value of threshold
	
	
	"""
	print "MQJLC?SF%PM%", path
	#create the image
	im = cv2.imread(path+'/'+fi)
	rows,cols,junk = im.shape
	#Verify the image shape
	if cols < rows:
		print('Reject ', fi, 'wrong orientation')
		return	
	if cols < 7000:
		print('Reject ',fi,' , too short for examination.')
		return
	
	#type of value	(np.uint -> 1-byte unsigned integer)
	#create the container for the two sides
	up_filled_blank= np.full((2*rows, 6980,3),0,np.uint8)
	down_filled_blank= np.full((2*rows, 6980,3),0,np.uint8)
	
	rowsBl, colsBl, junk = up_filled_blank.shape
	
	#place the images in the containers
	up_filled_blank[(rows+(rows+1)/2)-20:rowsBl, 0:colsBl]=im[0:rows/2+20,cols-6995: cols-15]
	down_filled_blank[0:((rows+1)/2)+20, 0:colsBl]=im[(rows/2)-20: rows, cols-6995: cols-15]
	
	#Flip the down image the be in the same way than the up image	
	down_filled_blank = np.flipud(down_filled_blank)
	
	
	if visual==1:
		cv2.namedWindow('Im', cv2.WINDOW_NORMAL)
		cv2.imshow('Im',im)
		cv2.namedWindow('UpFilledBlank', cv2.WINDOW_NORMAL)
		cv2.imshow('UpFilledBlank',up_filled_blank)
		cv2.namedWindow('DownFilledBlank', cv2.WINDOW_NORMAL)
		cv2.imshow('DownFilledBlank',down_filled_blank)
		k = cv2.waitKey(0) & 0xFF
		if k==27:
			cv2.destroyAllWindows()
	

	#Take the genreal shapes of the two sides
	thresh_Down, smooth_Down=smoother(down_filled_blank,visual,threshLevel)
	thresh_Up, smooth_Up=smoother(up_filled_blank,visual,threshLevel)
	
	#Flatten the two sides
	arrSideUp=calcSide(smooth_Up)
	arrSideDown=calcSide(smooth_Down)
	
	fi=fi.replace('.tiff','')
	flatThreshUp=flat(thresh_Up, arrSideUp, visual)
	cv2.imwrite(pathRes+'/'+fi+'Up.tiff', flatThreshUp)
	flatThreshDown=flat(thresh_Down,arrSideDown,visual)
	cv2.imwrite(pathRes+'/'+fi+'Down.tiff', flatThreshDown)

	
#Return the shape, the drop, of the side of a strip
def smoother(im, visual,threshLevel):
	"""
	Return an array of the shape of the strip's side
	
	Parameters
	----------
	*im: Side's strip image
	*visual: Enable to show the view of images or not
	*threshLevel: Level of the threshold
	
	Returns
	-------
	*thresh: The image's threshold
	*smooth: The general shape of the strip's side
	"""
	
	#A large brush to apply for getting a smooth shape
	kernel = np.ones((40,40),np.uint8)
	
	imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
	#Apply the threshold
	ret,thresh = cv2.threshold(imgray,threshLevel,255,cv2.THRESH_BINARY)
	#With this operation the strip loose all small details and become smooth
	thresh0 = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
	smooth = cv2.morphologyEx(thresh0, cv2.MORPH_CLOSE, kernel)
	
	if visual==1:
		cv2.namedWindow('Threshold', cv2.WINDOW_NORMAL)
		cv2.imshow('Threshold',thresh)
		cv2.namedWindow('Smoother', cv2.WINDOW_NORMAL)
		cv2.imshow('Smoother',smooth)
		k = cv2.waitKey(0) & 0xFF
		if k==27:
			cv2.destroyAllWindows()
			
	return thresh, smooth

def calcSide(smooth):
	"""
	Return an array of the depth where to take the column for making a linear strip.
	The depth correspond to 10 pixels below the smooth side.
	Parameter
	---------
	*smooth: the smooth strip's side
	
	Return
	------
	*arrSide: an array with the depth of the side where to take each column for making a flat side
	"""
	
	
	rows,cols= smooth.shape
	#Calculate how many non-zero number they are in each column
	arrSide = np.count_nonzero(smooth,axis=0)
	#Every result below 20 is set to 20
	arrSide = np.where(arrSide>20,arrSide,20)
	#Because the strip is in the down part of the image, we need to substract our numbers to the heigth
	#We adding 10 to don't be in middle of our border
	arrSide = rows - arrSide + 10
	return arrSide

#Return the side of a strip without drops
def flat(thresh, arrSide,visual):
	"""
	Return the flatten strip's side

	Parameters
	----------
	*thresh: thresholded image
	*arrside: array of depth for making the flat side
	
	Return
	------
	*flatThresh: the flatten thresholded image 
	"""
	
	rows, cols = thresh.shape
	#Create a container
	flatThresh= np.full((rows, cols), 0, np.uint8)
	
	for i in range(cols):
		#for each column we place in the container the corresponding column with the adequate depth 
		flatThresh[rows-arrSide[i]:rows, i:i+1]= thresh[0:arrSide[i], i:i+1]
	#Remove some unused full black rows
	flatThresh=flatThresh[rows-50:rows, 0:cols]
	
	if visual==1:
		print(path+'/ResultPreparation/flat.tiff')
		cv2.imwrite(path+'/ResultPreparation/flat.tiff', flatThresh)
		cv2.namedWindow('Flat', cv2.WINDOW_NORMAL)
		cv2.imshow('Flat',flatThresh)
		k = cv2.waitKey(0) & 0xFF
		if k==27:
			cv2.destroyAllWindows()
	
	return flatThresh
	
#Clearing the folder of past results	
def clearResultFolder(path):
	""""
	Delete all files in the given folder
	
	Parameter
	---------
	path: path of the folder to clearing
	"""
	filelist= os.listdir(path)
	for f in filelist:
		os.remove(path+'/'+f)
