# does a singular instance of insertion for the 0006R0.mp4

import natalialib as nlib
import numpy as np
import matplotlib.pyplot as plt;
import scipy.ndimage.filters as dconvolve;
from scipy.stats import norm;
import math;
# import time;
# import cv2;

rot_angle = 0;
video_name = '0006R0.mp4';
no_frames = 10;
freq_g = 4;
no_sds = 3;
patch_size = 21;

degree_increase = 15;
gphase = 0;
no_gframes = 7;


skip_convolution=0;

if skip_convolution==0:
	all_directions=nlib.get_all_dir(video_name,degree_increase,no_frames,freq_g, rot_angle, no_sds, patch_size, no_gframes);
	# do convolution and save
	np.savez('results_conv_dir',all_directions);
else:
	# load convolution
	with np.load('results_conv_dir.npz') as data:
		all_directions = data['arr_0'];

# get matrix with max motion
x = nlib.find_max_dir(all_directions);
# x.shape
# Out[81]: (10, 72, 96)
orientation = nlib.get_orientation(all_directions);

for c in range(0,no_frames):
 	plt.imshow(all_directions[orientation, c,:,:]);
# 	print('hi')

# load filter
with np.load('gFilterStack.npz') as data:
	g = data['arr_0'];
# g.shape
# Out[83]: (7, 21, 21)

maxcoors = (nlib.get_max_index(x));

frame_start=maxcoors[0]-np.round(g.shape[0]/2);
x1_start=maxcoors[1]-np.round(g.shape[1]/2);
x2_start=maxcoors[2]-np.round(g.shape[2]/2);

# create a matrix of dimension x with only filter in correct spot
maxInsert = np.zeros((x.shape));
maxInsert[maxcoors] = 1;

# gaborInsert = dconvolve.convolve(maxInsert,g);


# gauss = nlib.create_stack(rot_angle, no_gframes, freq_g, no_sds, patch_size);

# nlib.show_video(gauss,7)

# gaussInsert = dconvolve.convolve(maxInsert, gauss);

ones = np.ones((x.shape));

with np.load('movie_bg.npz') as data:
			video = data['arr_0'];
# plt.imshow(gaussInsert[maxcoors[0],:,:])

if skip_convolution==0:
	gaborInsert = dconvolve.convolve(maxInsert,g);
	gauss = nlib.gauss_stack(no_gframes,no_sds,patch_size);
	gaussInsert = dconvolve.convolve(maxInsert, gauss);
	# do convolution and save
	np.savez('filter_stacks',gaussInsert,gaborInsert);
else:
	# load convolution
	with np.load('filter_stacks.npz') as data:
		gaussInsert = data['arr_0'];
		gaborInsert = data['arr_1'];

gaussInsert = gaussInsert/np.max(gaussInsert);
gaborInsert = (gaborInsert/np.max(np.abs(gaborInsert))+1*127);
insertion = (gaborInsert * gaussInsert) + (video * (ones - gaussInsert));

plt.imshow(insertion[maxcoors[0],:,:]);

nlib.show_video(np.uint8(video),no_frames);
nlib.show_video(np.uint8(insertion*255), no_frames);
nlib.show_video(np.uint8(insertion), no_frames);