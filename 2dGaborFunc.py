
# import matplotlib.pyplot as plt
# import numpy as np
#from scipy.stats import norm
#import math
#from PIL import Image
import cv2
import natalialib as nlib;
#import time;
#import scipy.ndimage.filters as dconvolve;

video_name='06R0_labelled.mp4';
degree_increase=15;
no_frames=10;
freq_g=4;
# no_dir=(360/degree_increase) +1;

# group_frames=nlib.motionEnergyDefault(0,video_name,no_frames,freq_g);
# count = 0;
# all_directions=np.zeros((no_dir,no_frames,group_frames[0].shape[0],group_frames[0].shape[1]));
# for rot_angle in range (0,361,degree_increase):
#     all_directions[count,:,:,:]=nlib.motionEnergyDefault(rot_angle,video_name,no_frames,freq_g);
#     count+=1;

all_directions=nlib.find_max_dir(video_name,degree_increase,no_frames,freq_g)

imtoshow=all_directions[1,1,35,35];
print(imtoshow)
#    prints out 4 

# want to find max of one and min of second that correspond to each other
max_of_first=np.amax(first)

# n1=nlib.motionEnergyDefault(0,'06R0_labelled.mp4',10,4)
# n2=nlib.motionEnergyDefault(90,'06R0_labelled.mp4',10,4)

# ksize=21;
# no_gframes=7;
# rotation_angle=90;
# freq_g=4;
#

# k=nlib.gaborPatch(ksize,rotation_angle,freq_g,0,3);
# k=nlib.motionEnergyDefault(90,'0006R0.mp4')

#imToShow = Image.fromarray(np.uint8(n1[1,:,:]))
#cv2.imshow(imToShow)

#plt.figure(1); 
#imtoshow=np.uint8(n1[1,:,:]);
#cv2.imshow('first frame',imtoshow);
#
#plt.figure(2);
#anothaone=np.uint8(n1[2,:,:]);
#cv2.imshow('second frame',anothaone);
#
#plt.figure(3);
#imtoshow1=np.uint8(n2[1,:,:]);
#cv2.imshow('first frame labelled',imtoshow1);
#
#plt.figure(4);
#anothaone1=np.uint8(n2[2,:,:]);
#cv2.imshow('second frame labelled',anothaone1);
#
#cv2.destroyAllWindows()

# # get input values
# patch_size=201;
# rot_angle=30;
# # not sure in what format this should be
# # outSizeMatrix = input("Enter the size of the output matrix: ")

# delta = 15
# # # this is probably going to be a random number right?
# # phaseOfSin = input("Enter phase of sine to start: ")

# # frequencyOfSin = input("Enter frequency of sine wave: ")

# # orientOfSin = input("Enter orientation of sine wave: ")

# # stdevGauss = input("Enter stdev of Gaussian: ")

# # plug input values into equations
# x, y = np.meshgrid(np.linspace(-4,4,patch_size), np.linspace(-4,4,patch_size))

# k = norm.pdf((x**2 + y**2)**0.5) *np.sin(y);

# M = cv2.getRotationMatrix2D((k.shape[1]/2,k.shape[0]/2),rot_angle,1);
# k = cv2.warpAffine(k,M,(k.shape[1],k.shape[0]));

# fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(6,10))

# ax1.imshow(k,extent=[0,100,0,1])
# ax1.set_title('Default')
# plt.figure(1)
# im1 = plt.imshow(k,extent=[0,100,0,1], aspect='auto')
# plt.grid(True)

# plt.show()

# phase_set=np.linspace(0,math.pi,no_gframes);

# gstack1=np.zeros((len(phase_set),k.shape[0],k.shape[1]));
# gstack2=np.zeros((len(phase_set),k.shape[0],k.shape[1]));

# # fig, stack = plt.subplots(1,gstack.shape[0]);

# plt.figure(1);
# img_to_show=np.zeros((k.shape[0],k.shape[1],3));
# for phase_p in range(0,len(phase_set)):
# 	gstack1[phase_p,:,:]=nlib.gaborPatch(ksize,rotation_angle,freq_g,phase_set[phase_p],3);
# 	gstack2[phase_p,:,:]=nlib.gaborPatch(ksize,rotation_angle,freq_g,math.pi/2+phase_set[phase_p],3);
# 	img_to_show=np.concatenate((gstack1[phase_p,:,:],gstack2[phase_p,:,:]),1);
# 	# img_to_show[:,:,2]=100*gstack1[phase_p,:,:]+125;
# 	# img_to_show[:,:,1]=100*gstack2[phase_p,:,:]+125;
# 	# img_to_show=np.uint8(img_to_show/np.max(np.abs(img_to_show))*125+125);
# 	# img_to_show=np.uint8(img_to_show);
# 	cv2.imshow('frame', img_to_show);
# 	# plt.imshow(gstack[phase_p,:,:],extent=[0,100,0,1], aspect='auto');
# 	# plt.show();
# 	# stack[phase_p].imshow(gstack[phase_p,:,:],extent=[0,100,0,1], aspect='auto');
# 	if cv2.waitKey(1) & 0xFF == ord('q'):
# 		break;
# 	time.sleep(0.05);

# cv2.destroyAllWindows()

# plt.show();


# plt.set_title('Auto aspect')

# ax3.imshow(k,extent=[0,100,0,1], aspect=100)
# ax3.set_title('Manual aspect')

# filStack = [im1]
# # make stack of images
# fig, stack = plt.subplots(1,3)

# # while there are more frames to read in
# #or for the number of frames if we know how many
# for i in range(3):
# 	k = norm.pdf((x**2 + y**2)**0.5 *np.sin(x + delta*i))
# 	stack[i].imshow(k,extent=[0,100,0,1], aspect='auto')
# 	filStack.append(plt.imshow(k,extent=[0,100,0,1], aspect='auto'))
# plt.show()

# so then in figStack we have the stack of filters
# no_Frames = 40;
# # read in images from Camvid databases

# cap = cv2.VideoCapture('0006R0.mp4');
# ret, frame = cap.read();
# frame=cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),(288,216));
# movie_rec=np.zeros((no_Frames,frame.shape[0],frame.shape[1]));
# for frame_no in range(0,no_Frames):
#     ret, frame = cap.read();
#     movie_rec[frame_no,:,:]=cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),(288,216));
    
    
    

# g1=dconvolve.convolve(movie_rec,gstack1);
# g2=dconvolve.convolve(movie_rec,gstack2);
# g=np.power(g1,2)+np.power(g2,2);
# g=g/np.max(g);
# for frame_no in range(0,g.shape[0]):
#     imgtoshow=np.concatenate((movie_rec[frame_no,:,:],g[frame_no,:,:]*255),1);
#     imgtoshow=np.uint8(imgtoshow);
#     cv2.imshow('frame',imgtoshow);
#     if cv2.waitKey(1) & 0xFF == ord('q'):
# 		break;
#     time.sleep(0.05);
   
    
# cv2.destroyAllWindows()
# frame read correctly will make ret true
# ret, frame = cap.read()
# imageStack = []

# while(cap.isOpened()):
# 	# capture frame by frame
# 	# check if frame was read in correctly
# 	if ret == False:
# 		break
# 	# turn images to grayscale
# 	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# 	imageStack.append(gray)

# 	cv2.imshow('frame', gray)
# 	#the frame was correctly read
# 	ret, frame = cap.read()
# 	if cv2.waitKey(1) & 0xFF == ord('q'):
# 		break

# # make sure to release
# cap.release()


# print(imageStack.shape);

# for i in range(3):
# 	filterIm = filStack.pop()
# 	imageCon = imageStack.pop()
# 	for j in range(3):
# 		blurred_bw = np.convolve(filterIm[j], imageCon[j], mode='full')
# 		cv2.imshow('frame',blurred_bw)
