import natalialibNew as nlib;
import numpy as np;
import matplotlib.pyplot as plt;
import scipy.ndimage.filters as dconvolve;
from scipy.stats import norm;
import math;
# import time;
# import cv2;

rot_angle = 0;
# with np.load('Camvid_not_labelled.npz') as data:
# 	video_name = data['arr_0'];
video_name = 'Camvid_not_labelled.npz';
no_frames = 10;
freq_g = 4;
no_sds = 3;
patch_size = 21;

degree_increase = 15;
no_gframes = 7;


skip_convolution=1;

if skip_convolution==0:
	all_directions=nlib.get_all_dir(video_name,degree_increase,no_frames,freq_g, rot_angle, no_sds, patch_size, no_gframes);
	# do convolution and save
	np.savez('results_conv_dir',all_directions);
else:
	# load convolution
	with np.load('results_conv_dir.npz') as data:
		all_directions = data['arr_0'];

all_directions=all_directions[:,:,50:300,:];

# load filter
with np.load('gFilterStack.npz') as data:
	g = data['arr_0'];
# g.shape
# Out[83]: (7, 21, 21)

# gives spot in the array where there is the maximum energy
# orientation = (nlib.get_orientation(all_directions));

maxcoors = np.unravel_index(np.argmax(all_directions),all_directions.shape)
# get matrix with max motion
x = nlib.find_max_dir(all_directions);

# get the correct angle for filter
phase_set=np.linspace(0,math.pi*2,no_gframes);
rot_angle = maxcoors[0]*degree_increase;
for phase_p in range(0,len(phase_set)):
		g[phase_p,:,:]=nlib.gaborPatch(patch_size,rot_angle,freq_g,phase_set[phase_p],no_sds);


# frame_start=maxcoors[1]-np.round(g.shape[0]/2);
# x1_start=maxcoors[2]-np.round(g.shape[1]/2);
# x2_start=maxcoors[3]-np.round(g.shape[2]/2);

# create a matrix of dimension x with only filter in correct spot
maxInsert = np.zeros((x.shape));
maxInsert[maxcoors[1:4]] = 1;

# make an array of ones in the correct shape for insertion
ones = np.ones((x.shape));

# load the black and white original video 
with np.load('movie_bg.npz') as data:
	video = data['arr_0'];
	video=video[:,50:300,:];

# skip_convolution=1;
# convolution for the gaussian and gabor filters
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

# make sure gaussInsert ranges from 0-1
gaussInsert = gaussInsert/np.max(gaussInsert);
# make sure gabor ranges from 0-255
gaborInsert = (gaborInsert/np.max(np.abs(gaborInsert)+1)*127);
# gaborInsert = ((gaborInsert/np.max(np.abs(gaborInsert))*127)+127;
# add rgb channel and set one channel as what is being inserted and the other channel where its inserted
v=np.zeros((video.shape[0],video.shape[1],video.shape[2],3));
v[:,:,:,0]=(video * (ones - gaussInsert));
v[:,:,:,1]=(gaborInsert * gaussInsert);

# 'regular' smooth insertion
insertion = (gaborInsert * gaussInsert) + (video * (ones - gaussInsert));

# show the videos 
# show_video_big blows up the image size and slows down the video
nlib.show_video(np.uint8(video),5);
nlib.show_video(np.uint8(v), 5);
nlib.show_video(np.uint8(gaborInsert * gaussInsert), no_frames);


