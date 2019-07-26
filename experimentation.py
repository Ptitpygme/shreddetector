from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from mpl_toolkits.mplot3d import axes3d, Axes3D
from scipy.interpolate import griddata

import numpy as np
import cv2
import os
import copy
from tqdm import tqdm

def main():

	pathResExami='/udd/cvolantv/Pictures/ScanDetector/ResultEvaluation'

	filelist= os.listdir('/udd/cvolantv/Pictures/ScanDetector/ResultEvaluation')

	imDown=cv2.imread(pathResExami+'/ResCombineDown.tiff')
	imUp=cv2.imread(pathResExami+'/ResCombineUp.tiff')
	imUp=cv2.cvtColor(imUp,cv2.COLOR_BGR2GRAY)
	
	colorized(imUp,9)
'''
imDownResize = imDown[0:50 , 3000:4000]
imResize = imUp[30:45 , 3000:3600]



fig = plt.figure(figsize=(20,20))
ax = fig.gca(projection='3d')
print ax.set_xlim(xmin=0,xmax=300)
x= np.arange(600)
y= np.arange(15)
X,Y = np.meshgrid(x,y)

surf = ax.plot_surface(X, Y, imResize, cmap=cm.coolwarm, linewidth=0, antialiased=True)
ax.autoscale(enable=False)
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)


plt.show()
'''

def colorScale(shadeOfGray):
	res = []
	for i in range(shadeOfGray):
		B=255-i*2*(255/shadeOfGray)
		if(B<0): B=0 
		R=i*2*(255/shadeOfGray)-255
		if(R<0): R=0
		G=255-(B+R)
		res.append([B,G,R])
	return res

def funcColorize(im, scale):
	print im
	im = scale[im]
	
def colorized(im, shadeOfGray):
	scale = colorScale(shadeOfGray)
	vfun = np.vectorize(funcColorize)
	res = vfun(im, scale)
	
