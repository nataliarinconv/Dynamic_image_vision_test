import natalialibNew as nlib;
import numpy as np;
import matplotlib.pyplot as plt;
import scipy.ndimage.filters as dconvolve;
from scipy.stats import norm;
import math;


rot_angle = 0;
video_not_labelled = 'Camvid_not_labelled.npz';
video_labelled = 'Camvid_labelled.npz';
no_frames = 10;
freq_g = 4;
no_sds = 3;
patch_size = 21;
degree_increase = 15;
gphase = 0;
no_gframes = 7;


skip_convolution=1;

if skip_convolution==0:
	all_dir_not_labelled=nlib.get_all_dir(video_not_labelled,degree_increase,no_frames,freq_g, rot_angle, no_sds, patch_size, no_gframes);
	all_dir_labelled=nlib.get_all_dir(video_labelled,degree_increase,no_frames,freq_g, rot_angle, no_sds, patch_size, no_gframes);
	# do convolution and save
	np.savez('results_conv_dir',all_dir_not_labelled,all_dir_labelled);
else:
	# load convolution
	with np.load('results_conv_dir.npz') as data:
		all_dir_not_labelled = data['arr_0'];
		all_dir_labelled = data['arr_1'];

all_dir_not_labelled=all_dir_not_labelled[:,:,50:300,:];
all_dir_labelled=all_dir_labelled[:,:,50:300,:];

# load filter
with np.load('gFilterStack.npz') as data:
	g = data['arr_0'];
	g_l = data['arr_1'];
# get 4D coordinate for greatest 
maxcoors_not_labelled = nlib.get_max_index(all_dir_not_labelled);
maxcoors_labelled = nlib.get_max_index(all_dir_not_labelled);
# get matrix with max motion
max_not_l = nlib.find_max_dir(all_dir_not_labelled);
max_l = nlib.find_max_dir(all_dir_labelled);

# get the correct angle for filter
phase_set=np.linspace(0,math.pi*2,no_gframes);
rot_angle = maxcoors_not_labelled[0]*degree_increase;
rot_angle_l = maxcoors_labelled[0]*degree_increase;

for phase_p in range(0,len(phase_set)):
		g[phase_p,:,:]=nlib.gaborPatch(patch_size,rot_angle,freq_g,phase_set[phase_p],no_sds);
		g_l[phase_p,:,:]=nlib.gaborPatch(patch_size,rot_angle_l,freq_g,phase_set[phase_p],no_sds);

# GOTTA FIX THIS OR JUST CHECK IF ITS WORKING
# frame_start=maxcoors[1]-np.round(g.shape[0]/2);
# x1_start=maxcoors[2]-np.round(g.shape[1]/2);
# x2_start=maxcoors[3]-np.round(g.shape[2]/2);

# create a matrix of dimension x with only filter in correct spot
maxInsert = np.zeros((max_not_l.shape));
maxInsert[maxcoors_not_labelled[1:4]] = 1;

maxInsert_l = np.zeros((max_not_l.shape));
maxInsert_l[maxcoors_labelled[1:4]] = 1;

# make an array of ones in the correct shape for insertion
ones = np.ones((max_not_l.shape));

# load the black and white original video 
with np.load('bg_' + video_not_labelled) as data:
	video = data['arr_0'];
	video=video[:,50:300,:];
# load the black and white original video 
with np.load('bg_' + video_labelled) as data:
	video_l = data['arr_0'];
	video_l=video_l[:,50:300,:];

# skip_convolution=1;
# convolution for the gaussian and gabor filters
if skip_convolution==0:
	gaborInsert = dconvolve.convolve(maxInsert,g);
	gauss = nlib.gauss_stack(no_gframes,no_sds,patch_size);
	gaussInsert = dconvolve.convolve(maxInsert, gauss);

	gaborInsert_l = dconvolve.convolve(maxInsert_l,g_l);
	gaussInsert_l = dconvolve.convolve(maxInsert_l, gauss);	
	# do convolution and save
	np.savez('filter_stacks',gaussInsert,gaborInsert,gaussInsert_l,gaborInsert_l);
else:
	# load convolution
	with np.load('filter_stacks.npz') as data:
		gaussInsert = data['arr_0'];
		gaborInsert = data['arr_1'];
		gaussInsert_l = data['arr_2'];
		gaborInsert_l = data['arr_3'];

# make sure gaussInsert ranges from 0-1
gaussInsert = gaussInsert/np.max(gaussInsert);
gaussInsert_l = gaussInsert_l/np.max(gaussInsert_l);
# make sure gabor ranges from 0-255
gaborInsert = (gaborInsert/np.max(np.abs(gaborInsert)+1)*127);
gaborInsert_l = (gaborInsert_l/np.max(np.abs(gaborInsert_l)+1)*127);

# add rgb channel and set one channel as what is being inserted and the other channel where its inserted
v=np.zeros((video.shape[0],video.shape[1],video.shape[2],3));
v[:,:,:,0]=(video * (ones - gaussInsert));
v[:,:,:,1]=(gaborInsert * gaussInsert);

v_l=np.zeros((video_l.shape[0],video_l.shape[1],video_l.shape[2],3));
v_l[:,:,:,0]=(video_l * (ones - gaussInsert_l));
v_l[:,:,:,1]=(gaborInsert_l * gaussInsert_l);

# 'regular' smooth insertion
insertion = (gaborInsert * gaussInsert) + (video * (ones - gaussInsert));

# show the videos 
# show_video_big blows up the image size and slows down the video
nlib.show_video_big(np.uint8(video),no_frames);
nlib.show_video_big(np.uint8(v), no_frames);
nlib.show_video(np.uint8(gaborInsert * gaussInsert), no_frames);

nlib.show_video_big(np.uint8(video_l),no_frames);
nlib.show_video_big(np.uint8(v_l), no_frames);
nlib.show_video(np.uint8(gaborInsert_l * gaussInsert_l), no_frames);


