import natalialibNew as nlib;
import numpy as np;
import pickle;

skip_convolution = 1; 
pilot_1 = nlib.Natalia_object();

wavelets = nlib.make_noise(pilot_1);

if skip_convolution==0:
	motion_all_dir = nlib.get_all_dir(pilot_1);
	nlib.get_edges_labelled();
	# do convolution and save
	np.savez('results_conv_dir',motion_all_dir,pilot_1.labelled_edges);
else:
	# load convolution
	with np.load('results_conv_dir.npz') as data:
		motion_all_dir = data['arr_0'];
		pilot_1.labelled_edges = data['arr_1'];

# now we have all the max and the min for each frame
nlib.get_max_and_min_coors(pilot_1,motion_all_dir);

# picking which one we want function?
nlib.get_insertions(pilot_1);
# so now we have 2*pilot_1.no_frame amount of insertions so if we want 100 trials we need 50 frames

# to timestamp it --> save time.ctime(time.time())

file_pointer=open('pilot_1.pickle','wb');
pickle.dump(pilot_1,file_pointer);
file_pointer.close();



# file_pointer=open('filename.pickle','rb');
# pilot_1=pickle.load(file_pointer);
# file_pointer.close();