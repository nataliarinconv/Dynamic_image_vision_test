import cv2;
import math;
import numpy as np;
from scipy.stats import norm;
import matplotlib.pyplot as plt;
import time;
import scipy.ndimage.filters as dconvolve;

# Creates a 21x21 array. A singular frame of the gabor stack filter.
def gaborPatch(patch_size,rot_angle,freq_g,gphase,no_sds):
	x, y = np.meshgrid(np.linspace(-no_sds,no_sds,patch_size),\
		np.linspace(-no_sds,no_sds,patch_size));

	k = norm.pdf((x**2 + y**2)**0.5) *np.sin(y*math.pi/no_sds*freq_g+gphase);

	M = cv2.getRotationMatrix2D((k.shape[1]/2,k.shape[0]/2),rot_angle,1);
	k = cv2.warpAffine(k,M,(k.shape[1],k.shape[0]));
	k=k/np.max(k);
    # save the filter
	# np.savez('gFilter', k);
	return k;
def gaussPatch(patch_size,no_sds):
	x, y = np.meshgrid(np.linspace(-no_sds,no_sds,patch_size),\
	np.linspace(-no_sds,no_sds,patch_size));

	gauss = norm.pdf((x**2 + y**2)**0.5);
	return gauss;

def gauss_stack(no_gframes, no_sds, patch_size):
	k = gaussPatch(patch_size, no_sds);
	# no_gframes = 7;

	phase_set=np.linspace(0,math.pi,no_gframes);

	stack1=np.zeros((len(phase_set),k.shape[0],k.shape[1]));
	
    #creating filters here
	plt.figure(1);
	for phase_p in range(0,len(phase_set)):
		stack1[phase_p,:,:]=gaussPatch(patch_size,no_sds);
		
	# save the gstacks!
	# np.savez('gaussStack', gstack1);
	return  stack1;

def motion_energy(rot_angle, video_name, no_frames, freq_g, no_sds, patch_size, no_gframes):
	k = gaborPatch(patch_size, rot_angle, freq_g, 0, no_sds);

	phase_set=np.linspace(0,math.pi,no_gframes);

	gstack1=np.zeros((len(phase_set),k.shape[0],k.shape[1]));
	gstack2=np.zeros((len(phase_set),k.shape[0],k.shape[1]));

    #creating filters here
	plt.figure(1);
	for phase_p in range(0,len(phase_set)):
		gstack1[phase_p,:,:]=gaborPatch(patch_size,rot_angle,freq_g,phase_set[phase_p],no_sds);
		gstack2[phase_p,:,:]=gaborPatch(patch_size,rot_angle,freq_g,math.pi/2+phase_set[phase_p],no_sds);

	# save the gstacks!
	np.savez('gFilterStack', gstack1, gstack2);
	# read in images from Camvid databases

	cap = cv2.VideoCapture(video_name);
	ret, frame = cap.read();
	frame=cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),(480,360));
	movie_rec=np.zeros((no_frames,frame.shape[0],frame.shape[1]));
	for frame_no in range(0,no_frames):
	    ret, frame = cap.read();
	    movie_rec[frame_no,:,:]=cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),(480,360));
	     
	# save movie_rec
	np.savez('movie_bg', movie_rec);

	g1=dconvolve.convolve(movie_rec,gstack1);
	g2=dconvolve.convolve(movie_rec,gstack2);
	g=np.power(g1,2)+np.power(g2,2);
	g=g/np.max(g);
	    
	cv2.destroyAllWindows()
	cap.release()

	# save convolution
	# np.savez('convolution', np.uint8(g[:,:,:]*255)); 
	return  np.uint8(g[:,:,:]*255);

def get_all_dir(video_name,degree_increase,no_frames,freq_g, rot_angle, no_sds, patch_size, no_gframes):
	no_dir=(180/degree_increase) +1;

	group_frames=motion_energy(rot_angle,video_name,no_frames,freq_g,no_sds, patch_size, no_gframes);
	count = 0;
	all_directions=np.zeros((no_dir,no_frames,group_frames[0].shape[0],group_frames[0].shape[1]));

	for rot_ang in range (0,180,degree_increase):
	    all_directions[count,:,:,:]=motion_energy(rot_ang,video_name,no_frames,freq_g, no_sds, patch_size, no_gframes);
	    count+=1;

	# save all directions
	# np.savez('allDir',all_directions);
	return all_directions;

def find_max_dir(all_directions):
	return np.amax(all_directions, axis=0);

def get_max_index(matrix):
	return np.unravel_index(np.argmax(matrix, axis=None), matrix.shape);

# def get_orientation(all_directions):
# 	k = np.unravel_index(np.argmax(all_directions, axis=None), all_directions.shape);
# 	return k[0];

# def insert_filter(gFilter, movie, max_index):

def read_in_video(video_name):
	cap = cv2.VideoCapture(video_name);
	ret, frame = cap.read()

	full_video = [frame];
	while(cap.isOpened()):
			#Capture frame-by-frame	        
	        if ret == False:
	        	# print('Unable to read in frame');
	        	break
		#correctly read the frame 
		ret, frame = cap.read()
		full_video.append(frame);

	cv2.destroyAllWindows()
	cap.release()

	return full_video;

# def read_in_CamVid_images(labelled):
# 	start = 7959;

# 	if labelled==1:
# 		first = '0016E5_0' num2str(start) '_L.png';
# 	else:
# 		first =  '0016E5_0' num2str(start) '.png';

	

# 	return full_video;

def show_video(video_frames, no_frames):
	for no_fram in range (0,no_frames):
		cv2.imshow('frame', video_frames[no_fram]);

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
		time.sleep(0.1);

	cv2.destroyAllWindows()
