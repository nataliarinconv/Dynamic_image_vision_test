import natalialib as nlib
import numpy as np

video_lab='06R0_labelled.mp4';
video_reg='0006R0.mp4';

degree_increase=15;
no_frames=10;
freq_g=4;
skip_convolution=0;

# if skip_convolution==0:
# 	do convolution and save;
# else:
# 	load convolution;

# bestdir = nlib.get_best_dir(video_reg,degree_increase,no_frames,freq_g);

#max direction found for whatever number of frames
# # therefore here first is a 10,72,96 array
# first = nlib.find_max_dir(video_reg,degree_increase,no_frames,freq_g);
# second = nlib.find_max_dir(video_lab,degree_increase,no_frames,freq_g);

# # first_ascending = np.sort(first, axis=None);

# # make a structured array
# gtype = [('frame_no','uint8'), ('height', 'uint8'), ('width', 'uint8')];
# first_struct = np.array(first,dtype=gtype);
# second_struct = np.array(second,dtype=gtype);

# first_ascending = np.sort(first_struct, order=['height','width']);
# second_ascending = np.sort(second_struct, order=['height','width']);

# first_ascending[0,1,1];
# print('hey dude')
#first_top_max = first_ascending[]
# first_max = 0;
# second_max = 0;
# for frame_no in range (0,no_frames):
# 	new_first = np.amax

x = get_max_index(nlib.find_max_dir)