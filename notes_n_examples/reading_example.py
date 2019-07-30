import re;
import numpy as np;
import natalialibNew as nlib;
import scipy.ndimage.filters as dconvolve;
import matplotlib.pyplot as plt;


no_frame = 10;
no_labels = 32;

with np.load('Camvid_labelled.npz') as data:
	labelled = data['arr_0'];

file_colors = open("label_colors.txt","r");
# file_colors.read();
my_colors = file_colors.readlines();

matches = np.zeros((3,labelled.shape[1],labelled.shape[2]));
matches_ensemble = np.zeros((no_labels, no_frame,labelled.shape[1],labelled.shape[2]));

for count in range (0,32):
	# c will be one specific label Ex(64,128,64) which is animal
	c = re.findall('\d+', my_colors[count]);
	c = [int(c[2]),int(c[1]),int(c[0])];
	# c = np.tile(np.array(c).reshape((1,1,3)),(labelled.shape[1],labelled.shape[2],1));

	for fram_no in range(0,no_frame):
		frame = labelled[fram_no,:,:,:];
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
conv = np.zeros((fully_labelled.shape));

# for edge detection with the images
for fram in range(0,no_frame):
	# f is one of the frames
	f = fully_labelled[fram,:,:];
	conv[fram,:,:] = dconvolve.convolve(f,edge_mask);
	conv[fram,:,:] = dconvolve.convolve(conv[fram,:,:],thicker_mask);
	conv[fram,:,:] = np.abs(conv[fram,:,:])>0;


plt.imshow(conv[0,:,:])
nlib.show_video(conv,no_frame);

np.savez('labelled_edges',conv);