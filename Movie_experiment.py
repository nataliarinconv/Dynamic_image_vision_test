import natalialib as nlib;
import numpy as np;

class Movie_experiment:
	def _init_(self,video_name,video_labelled,no_frame,target_vs_nontarget,degree_increase,
		patch_size,freq_g,no_sds,no_gframes,no_directions,signal_amp,signal_p):

		self.video_name = video_name;
		self.video_labelled = video_labelled;
		self.no_frame = no_frame;
		self.target_vs_nontarget = target_vs_nontarget;
		self.degree_increase = degree_increase;
		self.patch_size = patch_size;
		self.freq_g = freq_g;
		self.no_sds = no_sds;
		self.no_gframes = no_gframes;
		self.no_directions = no_directions;
		self.signal_amp = signal_amp;
		self.signal_p = signal_p;
		self.rot_angle = 0;
		self.no_labels = 32;
		self.width = 960;
		self.height = 720;
		self.labelled = np.zeros((self.no_frame,self.height,self.width));

	def get_edges_labelled(self):

		labelled_o = lnib.get_labelled(self.video_labelled);

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
		conv = np.zeros((fully_labelled.shape));
		# for edge detection with the images
		for fram in range(0,self.no_frame):
			# f is one of the frames
			f = fully_labelled[fram,:,:];
			conv[fram,:,:] = dconvolve.convolve(f,edge_mask);
			conv[fram,:,:] = dconvolve.convolve(conv[fram,:,:],thicker_mask);
			conv[fram,:,:] = np.abs(conv[fram,:,:])>0;

		plt.imshow(conv[0,:,:])
		nlib.show_video(conv,self.no_frame);
		self.labelled = conv;


