import cv2;
import math;
import numpy as np;
from scipy.stats import norm;
import matplotlib.pyplot as plt;
import time;
import scipy.ndimage.filters as dconvolve;
import random;
from os import listdir;
from PIL import Image as PImage;
import re;
import pickle;

class Natalia_object:
	def __init__(self):
		self.no_trials = 20;
		self.video_name = 'Camvid_not_labelled.npz';
		self.video_labelled = 'Camvid_labelled.npz';
		self.type_of_vid = 2;
		self.no_frame = self.no_trials/2;
		self.target_vs_nontarget = 1;
		self.degree_increase = 15;
		self.patch_size = 21;
		self.freq_g = 4;
		self.no_sds = 3;
		self.no_gframes = 7; 
		self.no_directions = 16;
		self.signal_amp = 5;
		self.signal_p = 0;
		self.rot_angle = 0;
		self.no_labels = 32;
		self.width = 960;
		self.height = 720;
		self.width_resize =  self.width / 2; #480 
		self.height_resize =  self.height / 2;	#360	
		self.resize_lim_l = 50;
		self.resize_lim_u = 300;
		self.labelled_edges = np.zeros((self.no_frame,self.height_resize,self.width_resize));
		self.rgb_chan = 3;
		self.not_labelled = np.zeros((self.no_frame,self.height,self.width,self.rgb_chan));
		self.amplitude = np.zeros((self.no_directions));
		self.phasev = np.zeros((self.no_directions));
		self.max_indexes = [];
		self.min_indexes = []; 
		self.seed = [];
		self.the_time = [];
		self.max_insertion = [];
		self.min_insertion = [];

def run_block(self,skip_convolution):	
	# get the time
	self.the_time.append(time.ctime(time.time()));

	if skip_convolution==0:
		motion_all_dir = get_all_dir(self);
		get_edges_labelled();
		# do convolution and save
		np.savez('results_conv_dir',motion_all_dir,self.labelled_edges);
		mode = "w";
	else:
		mode = "a";
		# load convolution
		with np.load('results_conv_dir.npz') as data:
			motion_all_dir = data['arr_0'];
			self.labelled_edges = data['arr_1'];

	# now we have all the max and the min for each frame
	get_max_and_min_coors(self,motion_all_dir);
	for t in range(0,self.no_frame):
		get_insertions(self,t);
		# now we have a max and a min with every iteration

		# save seed,amplitude,phasev
		save_sap(self,mode);
	# choose which ones to show and show them

def save_sap(self,mode):
	# probably better with one file holding all the arrays together
	seeds = open("seeds.txt",mode);
	seeds.write(str(self.seed) + "\n");
	seeds.close();

	amps = open("amplitude.txt",mode);
	amps.write(str(self.amplitude) + "\n");
	amps.close();

	phase = open("phasev.txt",mode);
	phase.write(str(self.phasev) + "\n");
	phase.close();

def get_insertions(self,frames):
	# each time this is called so is get_rand_wavelets
	wavelets = make_noise(self);

	max_coors = self.max_indexes[frames];
	min_coors = self.min_indexes[frames];
	# this is how much rotation we need
	rot_max = (max_coors[0]) * self.degree_increase;
	rot_min = (min_coors[0]) * self.degree_increase;

	#rotate wavelets by the necessary rotation
	wavelets_max_dir = rotate_wavelets(self,wavelets,rot_max);
	wavelets_min_dir = rotate_wavelets(self,wavelets,rot_min);
	# inserted matrices
	self.max_insertion.append(insert_probe(self,max_coors,wavelets_max_dir));
	self.min_insertion.append(insert_probe(self,min_coors,wavelets_min_dir));

def get_max_and_min_coors(self,all_dir_not_labelled):
	for frame in range(0,self.no_frame):
		edge_frame = self.labelled_edges[frame,:,:];
		edge_frame = cv2.resize(edge_frame,(self.width_resize,self.height_resize));
		# isolate each frame in all directions
		frame_with_dirs = all_dir_not_labelled[:,frame,:,:];
		edge_frame = np.tile(edge_frame.reshape((1,self.height_resize,self.width_resize)),(frame_with_dirs.shape[0],1,1));
		frame_with_dirs1 = frame_with_dirs * (1 - edge_frame);
		# find max intensity coordinate for each frame
		frame_with_dirs2 = frame_with_dirs * edge_frame;
		frame_with_dirs2 = 1 - frame_with_dirs2;

		max_coors = get_max_index(frame_with_dirs1);
		min_coors = get_min_index(frame_with_dirs2);
		
		self.min_indexes.append((min_coors[0],frame,min_coors[1],min_coors[2])); 
		self.max_indexes.append((max_coors[0],frame,max_coors[1],max_coors[2])); 

def get_rand_vectors(self):
	target_shape = np.zeros((self.no_directions));
	target_shape[0]=self.signal_amp;
	nontarget_shape = np.zeros((self.no_directions));
	nontarget_shape[self.no_directions/2]=self.signal_amp;

	for d in range(0,self.no_directions):
		self.seed.append(time.time());			
		random.seed(self.seed[d]);
		x = random.gauss(1,1.0/3);
		x = x*(x>0 and x<=2)+2*(x>2);
		self.amplitude[d]=x;
		self.phasev[d] = random.uniform(0,math.pi*2);

	self.phasev[0] = self.signal_p;
	if self.target_vs_nontarget ==0:
		self.amplitude=self.amplitude+target_shape;
	else:
		self.amplitude=self.amplitude+nontarget_shape;

def get_edges_labelled(self):

	labelled_o = get_labelled(self.video_labelled);

	file_colors = open("label_colors.txt","r");
	my_colors = file_colors.readlines();
	matches = np.zeros((3,labelled_o.shape[1],labelled_o.shape[2]));
	matches_ensemble = np.zeros((self.no_labels, self.no_frame,labelled_o.shape[1],labelled_o.shape[2]));

	for count in range (0,self.no_labels):
		# c will be one specific label Ex(64,128,64) which is animal
		c = re.findall('\d+', my_colors[count]);
		c = [int(c[2]),int(c[1]),int(c[0])];
		# c = np.tile(np.array(c).reshape((1,1,3)),(labelled.shape[1],labelled.shape[2],1));

		for fram_no in range(0,self.no_frame):
			frame = labelled_o[fram_no,:,:,:];
			# y=(np.sum(frame == c,2)==3);
			# matches_ensemble[count,fram_no,:,:] = y; 
			if c in frame:
			# 	print(my_colors[count]);
			# 	print(count)
				for color_chan in range(0,3):
					# makes array of 0 & 1 where a match is found
					matches[color_chan] = ((frame[:,:,color_chan] == c[color_chan]) == True) * 1;
			# add up the different channels and divide by 3 so only the ones that == 3 are left 
			# // for floor division
			matches_ensemble[count,fram_no] = ((matches[0] + matches[1] + matches[2]) // 3) * count; 
		
	# now matches ensemble holds the newly labelled frames without rgb channel
	file_colors.close();
	fully_labelled = np.sum(matches_ensemble,0);
	edge_mask = -1.0/8*np.ones((3,3));
	edge_mask[1,1] = 1;
	thicker_mask = np.ones((3,3));
	# print(fully_labelled.shape)
	conv = np.zeros((self.no_frame,self.height_resize,self.width_resize));
	# for edge detection with the images
	for fram in range(0,self.no_frame):
		# f is one of the frames
		f = fully_labelled[fram,:,:];
		f = cv2.resize(f,(self.width_resize,self.height_resize));
		conv[fram,:,:] = dconvolve.convolve(f,edge_mask);
		conv[fram,:,:] = dconvolve.convolve(conv[fram,:,:],thicker_mask);
		conv[fram,:,:] = np.abs(conv[fram,:,:])>0;

	# plt.imshow(conv[0,:,:])
	# show_video(conv,self.no_frame);
	self.labelled_edges = conv;

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

	phase_set=np.linspace(0,math.pi*2,no_gframes);

	stack1=np.zeros((len(phase_set),k.shape[0],k.shape[1]));
	
    #creating filters here
	plt.figure(1);
	for phase_p in range(0,len(phase_set)):
		stack1[phase_p,:,:]=gaussPatch(patch_size,no_sds);
		
	# save the gstacks!
	# np.savez('gaussStack', gstack1);
	return  stack1;

def motion_energy(self):
	k = gaborPatch(self.patch_size, self.rot_angle, self.freq_g, self.rot_angle, self.no_sds);

	phase_set=np.linspace(0,math.pi*2,self.no_gframes);

	gstack1=np.zeros((len(phase_set),k.shape[0],k.shape[1]));
	gstack2=np.zeros((len(phase_set),k.shape[0],k.shape[1]));

    #creating filters here
	plt.figure(1);
	for phase_p in range(0,len(phase_set)):
		gstack1[phase_p,:,:]=gaborPatch(self.patch_size,self.rot_angle,self.freq_g,phase_set[phase_p],self.no_sds);
		gstack2[phase_p,:,:]=gaborPatch(self.patch_size,self.rot_angle,self.freq_g,math.pi/2+phase_set[phase_p],self.no_sds);

	# save the gstacks!
	np.savez('gFilterStack', gstack1, gstack2);
	# read in images from Camvid databases

	if self.type_of_vid == 1:
		# for moving line
		with np.load('moving_line.npz') as data:
			movie_rec = data['arr_0'];
	if self.type_of_vid == 2:
		# for camvid
		with np.load(self.video_name) as data:
			movie_r = data['arr_0'];
		movie_rec=np.zeros((self.no_frame,self.height_resize,self.width_resize));

		for no_fram in range(0,self.no_frame):
			img = movie_r[no_fram,:,:,:];
			img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),(self.width_resize,self.height_resize));

			movie_rec[no_fram,:,:] = img;
	else:
		# for the regular file
		cap = cv2.VideoCapture(self.video_name);
		ret, frame = cap.read();
		frame=cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),(self.width_resize,self.height_resize));
		movie_rec=np.zeros((self.no_frame,frame.shape[0],frame.shape[1]));
		for frame_no in range(0,self.no_frame):
		    ret, frame = cap.read();
		    movie_rec[frame_no,:,:]=cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),(self.width_resize,self.height_resize));
		     
	movie_bg = 'bg_' + self.video_name;     
	# save movie_rec
	np.savez(movie_bg, movie_rec);

	g1=dconvolve.convolve(movie_rec,gstack1);
	g2=dconvolve.convolve(movie_rec,gstack2);
	g=np.power(g1,2)+np.power(g2,2);
	# g=g/np.max(g);
	    
	# cv2.destroyAllWindows()
	# cap.release()

	# save convolution
	# np.savez('convolution', np.uint8(g[:,:,:]*255)); 
	return  g;

# video_name,degree_increase,no_frame,freq_g, rot_angle, no_sds, patch_size, no_gframes
def get_all_dir(self):
	no_dir=(self.height_resize/self.degree_increase);

	group_frames=motion_energy(self);
	count = 0;
	all_directions=np.zeros((no_dir,self.no_frame,group_frames[0].shape[0],group_frames[0].shape[1]));

	for rot_ang in range (0,self.height_resize,self.degree_increase):
	    all_directions[count,:,:,:]=motion_energy(self);
	    count+=1;

	# save all directions
	# np.savez('allDir',all_directions);
	all_directions=all_directions-np.min(all_directions);
	all_directions=all_directions/np.max(all_directions);
	all_directions=np.uint8(all_directions*255);
	return all_directions;

def get_orientation(all_directions):

	mean_intensities = np.zeros((all_directions.shape[0])); #just an array of directions?
	# get the average across the ten frames for each direction
	for d in range(0,all_directions.shape[0]): #across every direction
		for f in range(0,all_directions.shape[1]): #across each frame in each direction
			mean_intensities[d] = np.mean(all_directions[d,f,:,:]);

	# find the max out of those and now we have the direction we want
	max_int = np.amax(mean_intensities, axis=0);

	return  np.where(mean_intensities==max_int);

# def find_max_dir(maxcoors, all_directions):
# 	return all_directions[maxcoors,:,:,:];

def find_max_dir(all_directions):
	return np.amax(all_directions, axis=0);

def get_max_index(matrix):
	return np.unravel_index(np.argmax(matrix, axis=None), matrix.shape);

def get_min_index(matrix):
	return np.unravel_index(np.argmin(matrix, axis=None), matrix.shape);

# def loadImages(path):
#     # return array of images
#     imagesList = sorted(listdir(path));
#     loadedImages =[];

#     for image in imagesList:
#         img = cv2.imread(path + image);
#         loadedImages.append(img);

#     return loadedImages;

def read_in_video(self):
# 	if self.video_name.find('Camvid') > 0:
# 		path = "./not_labelled/";
# 		self.not_labelled = load_images(path);
# 	else:
	cap = cv2.VideoCapture(self.video_name);
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

# 		self.not_labelled = full_video;
	return full_video;

def show_video_big(video_frames, no_frame):
	for no_fram in range (0,no_frame):
		img = video_frames[no_fram];
		img = cv2.resize(img, (img.shape[1]*3,img.shape[0]*3));
		cv2.resizeWindow('frame', 665,500);
		cv2.imshow('frame', img);
		
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
		time.sleep(1);

	cv2.destroyAllWindows()

def show_video(video_frames, no_frame):
	for no_fram in range (0,no_frame):
		img = video_frames[no_fram];
		# img = cv2.resize(img, (img.shape[1]*3,img.shape[0]*3));
		# cv2.resizeWindow('frame', 665,500);
		cv2.imshow('frame', img);
		
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
		time.sleep(1);

	cv2.destroyAllWindows()

def make_stack(frame, no_frame, rgb_chan):
	vid = np.zeros((no_frame,frame.shape[0],frame.shape[1],frame.shape[2]));
	for f in range(0,no_frame):
		vid[f,:,:,:] = frame;
	return vid;
	# count = rgb_chan;
	# if count == 1:
	# 	vid = np.zeros((no_frame,frame.shape[0],frame.shape[1],frame.shape[2]));
	# 	for f in range(0,no_frame):
	# 		vid[f,:,:,:] = frame;
	# else:
	# 	vid = np.zeros((no_frame,frame.shape[0],frame.shape[1]));
	# 	for f in range(0,no_frame):
	# 		vid[f,:,:] = frame;

def get_labelled(video_name):
 	with np.load('Camvid_labelled.npz') as data:
		video_labelled = data['arr_0'];
	# if not the fully labelled set of images
	if video_name.find('Camvid') < 0:
		video_labelled = read_in_video(video_name)
	return video_labelled;

def make_noise(self):
	# make the random vectors
	get_rand_vectors(self);
	noise = np.zeros((self.no_directions,self.no_gframes,self.patch_size,self.patch_size));
	to_insert = np.zeros((self.no_directions,self.no_gframes,self.patch_size,self.patch_size));
	d = 0;
	# to make 16 different directions
	for rot in np.arange(0,self.height_resize,360.0/self.no_directions):
		# each one needs a random starting phase
		phase_set=np.linspace(0,math.pi*2,self.no_gframes)+self.phasev[d];
		for phase_p in range(0,len(phase_set)):
				noise[d,phase_p,:,:]=self.amplitude[d]*gaborPatch(self.patch_size,rot,self.freq_g,phase_set[phase_p],self.no_sds);
		d = d + 1;
	return np.sum(noise,0);

def rotate_wavelets(self, video,rot_angle):
	vid = np.zeros((video.shape[0],video.shape[1],video.shape[2]))
	for frame in range(0,self.no_gframes):
		k = video[frame,:,:];
		M = cv2.getRotationMatrix2D((k.shape[1]/2,k.shape[0]/2),rot_angle,1);
		k = cv2.warpAffine(k,M,(k.shape[1],k.shape[0])); 
		# k=k/np.max(k);
		vid[frame,:,:] = k;

	return vid;

def insert_probe(self,coors, g):
	# create a matrix of dimension x with only filter in correct spot
	# show_video(self.labelled_edges,5)
	edges = self.labelled_edges[:,self.resize_lim_l:self.resize_lim_u,:];
	mInsert = np.zeros((edges.shape));
	# print(edges.shape)
	# print(mInsert.shape)

	mInsert[coors[1:4]] = 1;
	# show_video_big(np.uint8(mInsert*255),self.no_frame)
	# make an array of ones in the correct shape for insertion
	ones = np.ones((edges.shape));
	# print(ones.shape)
	# load the black and white original video 
	with np.load('movie_bg.npz') as data:
		video = data['arr_0'];
		# show_video(np.uint8(video),5)
		video=video[:,self.resize_lim_l:self.resize_lim_u,:];
		# print(video.shape)
	# convolution for the gaussian and gabor filters

	gaborInsert = dconvolve.convolve(mInsert,g);
	gauss = gauss_stack(self.no_gframes,self.no_sds,self.patch_size);
	gaussInsert = dconvolve.convolve(mInsert, gauss);
	# print(gaussInsert.shape)
	# do convolution and save

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
	insertion =  (gaborInsert * gaussInsert) + (video * (ones - gaussInsert));

	# show the videos 
	# show_video_big blows up the image size and slows down the video
	# show_video_big(np.uint8(video),self.no_frame);
	# show_video_big(np.uint8(v), self.no_frame);
	# show_video_big(np.uint8(gaborInsert), self.no_frame);
	# show_video_big(np.uint8(gaborInsert * gaussInsert), self.no_frame);
	return insertion;
