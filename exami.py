import numpy as np
import cv2
import os


def preparation(path, pathRes,visual):
	clearResultFolder(pathRes)
	filelist= os.listdir(path)
	print filelist
	for fi in filelist:
		shape(path, fi, pathRes,visual)

#Create the images of differents sides of strip without drops
def shape(path, fi, pathRes,visual):
	im = cv2.imread(path+'/'+fi)
	rows,cols,junk = im.shape
	print fi
	print im.shape
	if cols < rows:
		print('Reject ', fi, 'wrong orientation')
		return	
	if cols < 6990:
		print('Reject ',fi,' , too short for examination.')
		return
	#np.full
	#Size (rows, cols)
	#Content value
	#type of value	(np.uint -> 1-byte unsigned integer)
	up_filled_blank= np.full((2*rows, 6980,3),0,np.uint8)
	down_filled_blank= np.full((2*rows, 6982,3),0,np.uint8)
	castIm = np.full((100, 6980),0,np.uint8)
	rowsBl, colsBl, junk = up_filled_blank.shape
	
	up = im[0:rows/2+20, 15: cols -(cols-6995)]
	#down = im[rows/2: rows, 15: cols -15]
	print im.shape
	print('Up: ',up.shape)
	up_filled_blank[(rows+(rows+1)/2)-20:rowsBl, 0:colsBl]=im[0:rows/2+20, 15: cols -(cols-6995)]
	down_filled_blank[0:((rows+1)/2)+20, 1:colsBl+1]=im[(rows/2)-20: rows, (cols-6995): cols-15]
	
		
	M = cv2.getRotationMatrix2D((colsBl/2,rowsBl/2),180,1)
	down_filled_blank = cv2.warpAffine(down_filled_blank,M,(colsBl,rowsBl))	
	
	if visual==1:
		cv2.namedWindow('Im', cv2.WINDOW_NORMAL)
		cv2.imshow('Im',im)
		#cv2.namedWindow('Up', cv2.WINDOW_NORMAL)
		#cv2.imshow('Up',up)
		#cv2.namedWindow('Down', cv2.WINDOW_NORMAL)
		#cv2.imshow('Down',down)
		cv2.namedWindow('UpFilledBlank', cv2.WINDOW_NORMAL)
		cv2.imshow('UpFilledBlank',up_filled_blank)
		cv2.namedWindow('DownFilledBlank', cv2.WINDOW_NORMAL)
		cv2.imshow('DownFilledBlank',down_filled_blank)
		k = cv2.waitKey(0) & 0xFF
		if k==27:
			cv2.destroyAllWindows()
	
	thresh_Down, smooth_Down=smoother(down_filled_blank,visual)
	thresh_Up, smooth_Up=smoother(up_filled_blank,visual)
	print 'Up'
	arrSideUp=calcSide(thresh_Up, smooth_Up)
	print 'Down'
	arrSideDown=calcSide(thresh_Down, smooth_Down)
	fi=fi.replace('.tiff','')
	flatThreshUp=flat(thresh_Up, arrSideUp, visual)
	
	
	
	cv2.imwrite('/udd/cvolantv/Pictures/ScanDetector/ResultPreparation/'+fi+'Up.tiff', flatThreshUp)
	flatThreshDown=flat(thresh_Down,arrSideDown,visual)
	cv2.imwrite('/udd/cvolantv/Pictures/ScanDetector/ResultPreparation/'+fi+'Down.tiff', flatThreshDown)

	
#Return the shape, the drop, of the side of a strip
def smoother(im, visual):
	
	kernel = np.ones((40,40),np.uint8)
	imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
	ret,thresh = cv2.threshold(imgray,180,255,cv2.THRESH_BINARY)
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

def calcSide(thresh,smooth):
	rows, cols = thresh.shape
	arrSide=[]
	for i in range(cols):
		nSmooth=0
		for j in range(rows):
			if smooth[j,i]==255:
				nSmooth=nSmooth+1
		if(nSmooth<11):
			print (i,'AAAAHHHHH',nSmooth)
			nSmooth=nSmooth+20
		
		arrSide.append((rows-nSmooth)+10)

	return arrSide

#Return the side of a strip without drops
def flat(thresh, arrSide,visual):
	rows, cols = thresh.shape
	flatThresh= np.full((rows, cols), 0, np.uint8)
	for i in range(len(arrSide)):
		flatThresh[rows-arrSide[i]:rows, i:i+1]= thresh[0:arrSide[i], i:i+1]
	flatThresh=flatThresh[rows-50:rows, 0:cols]
	if visual==1:
		cv2.imwrite('/udd/cvolantv/Pictures/ScanDetector/ResultPreparation/flat.tiff', flatThresh)
		cv2.namedWindow('Flat', cv2.WINDOW_NORMAL)
		cv2.imshow('Flat',flatThresh)
		k = cv2.waitKey(0) & 0xFF
		if k==27:
			cv2.destroyAllWindows()
	
	return flatThresh
	
#Clearing the folder of past results	
def clearResultFolder(path):
	filelist= os.listdir(path)
	print filelist
	for f in filelist:
		os.remove(path+'/'+f)
