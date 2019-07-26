import numpy as np





def heightFinder(im):
	"""
	Return the base and the top of bumps from a strip's side image
	
	Parameter
	---------
	im: strip's side in black and white
	
	Return 
	------
	1-the base of bumps
	2-the top of bumps
	"""
	rows,cols = im.shape
	i = rows-1	

	floor =-1
	rowsFloor = im[i]
	#Count the number of differents pixels
	unique, counts = np.unique(rowsFloor, return_counts=True)
	#if they are black and pixels
	if counts.size >1 :
		nbOcc = counts[1] 	
		prop = float(nbOcc)/cols
	else:
		#If they are only one type of pixels
		prop = 1.0
	#while we have more than 95% of white pixels
	while (prop > 0.95) & (i >= 0):
		rowsFloor = im[i]
		unique, counts = np.unique(rowsFloor, return_counts=True)
		if counts.size >1 :
			nbOcc = counts[1] 	
			prop = float(nbOcc)/cols
		else:
			prop=1.0
		floor=i
		i-=1
		
	#Repeat operation for 5%  of white pixels
	rowsTop=im[i]
	top=-1
	while (prop > 0.05) & (i >= 0):
		rowsRoof = im[i]
		unique, counts = np.unique(rowsRoof, return_counts=True)
		if counts.size >1 :
			nbOcc = counts[1] 	
			prop = float(nbOcc)/cols
		else:
			prop=1.0
		top=i
		i-=1
	
	
	return floor, top
