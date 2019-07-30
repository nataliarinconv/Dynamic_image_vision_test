import natalialibNew as nlib
import numpy as np
import matplotlib.pyplot as plt;
import cv2;
import scipy.ndimage.filters as dconvolve;

no_frames = 20;
height = 230;
width = 240;
no_frames = 10;
height = 360;
width = 480;
x1 = 100;
x2 = 101;
length = range(100,200);
rot_angle = 30;

size = [no_frames,height,width];
moving_line = np.zeros((size));


moving_line[0,length,x1:x2] = 1;

for f in range(1,no_frames):
	moving_line[f-1,length,x1:x2] = 0;
	moving_line[f,length,x1:x2] = 1;
	x1+=1;
	x2+=1;

for f in range(0,no_frames):
	k = moving_line[f,:,:];
	M = cv2.getRotationMatrix2D((k.shape[1]/2,k.shape[0]/2),rot_angle,1);
	k = cv2.warpAffine(k,M,(k.shape[1],k.shape[0]));
	moving_line[f,:,:] = k;



nlib.show_video(moving_line,no_frames);

np.savez('moving_line', moving_line);