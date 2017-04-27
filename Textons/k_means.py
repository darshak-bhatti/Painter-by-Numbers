import numpy as np
from datetime import datetime
import pickle
import scipy
import glob
from PIL import Image
from sklearn.metrics.pairwise import pairwise_distances
from multiprocessing import Pool
from numpy.core.umath_tests import inner1d
import random
from multiprocessing import Pool
import time

random.seed(1)

train_files = [f for f in glob.glob("patches2/*")]
#np.set_printoptions(threshold=np.nan)


def flatten(file):
	temp_features_flattened = []

	jpgfile = Image.open(file)
	t = np.asarray(jpgfile)
	temp_features_flattened.append(t.flatten())
	return np.array(temp_features_flattened)[0]

def populate_features():
	features = []
	features_flattened = []
	
	print len(train_files)
	thread_count = 20

	subset_size = len(train_files[:])
	patch_dim = 35
	pool = Pool(thread_count)
	features = pool.map(flatten, train_files[:subset_size])
	pool.close()  	
	pool.join() 

	return np.array(features)


features = populate_features()


def HausdorffDist(A,B):
	A = A.reshape(35,35)
	B = B.reshape(35,35)
	# Find pairwise distance
	D_mat = np.sqrt(inner1d(A,A)[np.newaxis].T + inner1d(B,B)-2*(np.dot(A,B.T)))
	# Find DH
	dH = np.max(np.array([np.max(np.min(D_mat,axis=0)),np.max(np.min(D_mat,axis=1))]))
	return(dH)

def make_new_centers(features, centers):
	to_return = []
	# print 'centres:', centers
	for x in centers:
		if len(x) == 0:
			continue
		temp_matrix = np.array([features[i] for i in x])
		# print temp_matrix.shape, '\n*********************************\n'
		to_return.append(np.mean(temp_matrix, axis = 0))


		# print 'Original:', temp_matrix, '\n##########\nMean:', np.array(to_return), '\n'
	# print np.array(to_return)
	return np.array(to_return)

def pairwise(inp):
	features = inp[0]
	init_centres = inp[1]
	return pairwise_distances(features, init_centres, metric=HausdorffDist)



def k_means(features):
	n_centres = 350
	init_centres_ind = []
	print 'here'

	i = 0
	while i < n_centres:
		print 'i:', i
		x = random.randint(0, len(features))
		if x not in init_centres_ind:
			init_centres_ind.append(x)
			i += 1


	
	init_centres = np.array([features[x] for x in init_centres_ind])
	print init_centres
	c = 0
	final_start = float(time.time())
	while c < 1000:
		# print 'c:', c
		start = float(time.time())

		#Multiprocessing to calculate distance of each point to each center
		thread_count = 60

		multiprocessing_arr = []
		print '\n', 'Shape:', features.shape, init_centres.shape

		step = features.shape[0]/thread_count
		for i in range(step):
			sub_features = features[i*thread_count:i*thread_count+thread_count,:]
			to_append = np.array([sub_features, init_centres])

			multiprocessing_arr.append(to_append)

		sub_features = features[(i+1)*thread_count:,:]
		if len(sub_features) > 0:
			to_append = np.array([sub_features, init_centres])
			multiprocessing_arr.append(to_append)

		multiprocessing_arr = np.array(multiprocessing_arr)

		pool = Pool(thread_count)
		res_mat = pool.map(pairwise, multiprocessing_arr)
		pool.close()  	
		pool.join() 

		# print res_mat, '\n'

		res_mat = np.array(res_mat)
		temp = res_mat[0]
		for e in res_mat[1:]:
			temp = np.append(temp, e)


		res_mat = temp.reshape(features.shape[0], n_centres)



		# res_mat = pairwise_distances(features, init_centres, metric=HausdorffDist)
		# print res_mat
		# print '^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^'
		# print np.array_equal(temp, res_mat)

		
		print 'time:', float(time.time()) - start


		count = 0
		centers = [[] for _ in range(n_centres)]
		for i, e in enumerate(res_mat):
			# print 'i:', i
			if min(e) == 0:
				count += 1
			centre_ind = np.where(e==min(e))[0][0]
			# print 'c:', c, 'centers', min(e), centre_ind, '\n'
			centers[centre_ind].append(i)
		new_centers = make_new_centers(features, centers)
		print 'c:', c, new_centers.shape
		# if new_centers.tolist() == init_centres.tolist():
		# print 'compare', new_centers, '\n'
		if np.array_equal(new_centers, init_centres):
			break
		init_centres = np.copy(new_centers)
		print init_centres, '#####################', '/n'

		# Dump the trained  with k_means_model Pickle
		model_filename = 'k_means_models/k_means_clustering_' + str(datetime.now()).split('.')[0] + '.pkl'
	
		# Open the file to save as pkl file
		model_pkl = open(model_filename, 'wb')
		pickle.dump(init_centres, model_pkl)
		# Close the pickle instances
		model_pkl.close()


		n_centres = new_centers.shape[0]

		c += 1

	if c >= 400:
		print 'Did Not Converge'
	print 'K_MEANS TIME:', float(time.time()) 
	- final_start
	return new_centers


model = k_means(features)
print model
print model.shape

# Dump the trained  with k_means_model Pickle
model_filename = 'k_means_clustering_350' + str(datetime.now()).split()[0] + '.pkl'
	
# Open the file to save as pkl file
model_pkl = open(model_filename, 'wb')
pickle.dump(model, model_pkl)
# Close the pickle instances
model_pkl.close()
