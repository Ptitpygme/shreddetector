import numpy as np





def heightFinder(im):
	rows,cols = im.shape
	i = rows-1	

	floor =-1
	rowsFloor = im[i]
	unique, counts = np.unique(rowsFloor, return_counts=True)
	if counts.size >1 :
		nbOcc = counts[1] 	
		prop = float(nbOcc)/cols
	else:
		prop = 1.0
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
