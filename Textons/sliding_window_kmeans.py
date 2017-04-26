from datetime import datetime
import pickle
import numpy as np
from PIL import Image, ImageFilter
from multiprocessing import Pool
import random
from numpy.core.umath_tests import inner1d
from sklearn.preprocessing import normalize
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import os, glob
import sys


random.seed(1)
model_pkl_filename = sys.argv[1]
model_pkl = open(model_pkl_filename, 'rb')
model = pickle.load(model_pkl)

n_clusters = model.shape[0]
patch_dim = 35
 
print model.shape

def labelled_data_dict(image_file):
	f = open(image_file, 'r')
	artist_to_paintings_dict = {}
	painting_to_artist_dict = {}
	for i,line in enumerate(f):
		painting, artist = line[:-1].split(',')[:2]
		painting_to_artist_dict[painting] = artist
		try:
			artist_to_paintings_dict[artist].append(painting)
		except Exception:
			artist_to_paintings_dict[artist] = [painting]
	return artist_to_paintings_dict, painting_to_artist_dict

artist_to_paintings_dict, painting_to_artist_dict = labelled_data_dict('train_info.csv')
print len(painting_to_artist_dict), len(artist_to_paintings_dict)

for e in artist_to_paintings_dict:
	if len(artist_to_paintings_dict[e]) > 120:
		selected = e
		to_process = artist_to_paintings_dict[e]
		break

extras = []
train_files = [f for f in glob.glob("patches2/*")]
for files in train_files:
	temp = files.split('.')[0][9:] + '.jpg'
	if temp not in to_process and temp not in extras:
		extras.append(temp)


print selected
print 'WOOOOOOOOOOORKKKKKKKKKKKKK'
print len(to_process)
print len(extras)
to_process = to_process+extras
print selected
print len(to_process)

def HausdorffDist(A,B):
	# Find pairwise distance
	D_mat = np.sqrt(inner1d(A,A)[np.newaxis].T + inner1d(B,B)-2*(np.dot(A,B.T)))
	# Find DH
	dH = np.max(np.array([np.max(np.min(D_mat,axis=0)),np.max(np.min(D_mat,axis=1))]))
	return(dH)

def predict(X):

	mini = float('inf')
	for i, e in enumerate(model):
		a = e.reshape(35,35)
		dist = HausdorffDist(a,X)
		if dist < mini:
			mini = dist
			to_return = i

	return to_return

count = 0

def sliding_window(inp):
	global count
	t = inp[0]
	label = inp[1]
	temp_texton = [0] * n_clusters
	x = 0 
	y = 0
	while y < t.shape[1]:
		if y + patch_dim > t.shape[1]:
			break
		x = 0
		while x < t.shape[0]:
			if x + patch_dim > t.shape[0]:
				break

			temp = t[x:x + patch_dim, y:y + patch_dim]
			prediction = predict(temp)
			print 'prediction', str(count), ':', prediction
			count += 1

			temp_texton[prediction] += 1

			x += patch_dim
		y += patch_dim

	return [temp_texton, label]



def main():
	thread_count = 56
	texton = []
	to_pass = []

	labels = []

	count = 0
	for painting in to_process:
		file = 'train/'+painting
		# labels.append(painting_to_artist_dict[painting])
		jpgfile = Image.open(file).convert('L')
		t = np.asarray(jpgfile)
		pos_label = painting_to_artist_dict[painting]
		if pos_label != selected:
			pos_label = 'not_artist'
		to_pass.append([t, pos_label])

	pool = Pool(thread_count)
	result = pool.map(sliding_window, to_pass)
	pool.close()  	
	pool.join() 
	
	random.shuffle(result)
	for e in result:
		texton.append(e[0])
		labels.append(e[1])

		# temp_texton = [0] * n_clusters
		# x = 0 
		# y = 0
		# while y < t.shape[1]:
		# 	if y + patch_dim > t.shape[1]:
		# 		break
		# 	x = 0
		# 	while x < t.shape[0]:
		# 		if x + patch_dim > t.shape[0]:
		# 			break

		# 		temp = t[x:x + patch_dim, y:y + patch_dim]
		# 		prediction = predict(temp)
		# 		print 'prediction', str(count), ':', prediction
		# 		count += 1

		# 		temp_texton[prediction] += 1

		# 		x += patch_dim
		# 	y += patch_dim

		# texton.append(temp_texton[:])
	return texton, labels

texton, labels = main()

#print texton
#print labels

np.set_printoptions(linewidth=175)

texton = np.array(texton)
print 'SHAPE:', texton.shape, len(labels)

#texton = normalize(texton, axis=0, norm='l1')
# print texton

subset_size = texton.shape[0]

train_features = np.array(texton[:int(0.8* subset_size),])
train_labels = np.array(labels[:int(0.8 * subset_size)])

test_features = np.array(texton[int(0.8 * subset_size):,])
test_labels = np.array(labels[int(0.8 * subset_size):])

#train_features = np.array(texton[:-2,])
#train_labels = np.array(labels[:-2])

#test_features = np.array(texton[-2:,])
#test_labels = np.array(labels[-2:])


print '\n\n######################################33'

# print train_features
print train_labels
 
print 'VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVvv'

# print test_features
print test_labels

#clf = SVC(kernel = 'poly', C = 1.5, degree = 3, random_state=0)
#clf.fit(train_features, train_labels)

#clf = DecisionTreeClassifier(random_state=0, min_samples_leaf=4)
#clf.fit(train_features, train_labels)

clf = RandomForestClassifier(random_state=0, n_estimators=12, n_jobs = -1, verbose = 10)
clf1 = AdaBoostClassifier(n_estimators = 50, learning_rate = 0.05, random_state = 0)
clf.fit(train_features, train_labels)
clf1.fit(train_features, train_labels) 

print 'Train Accuaracy:', accuracy_score(train_labels, clf.predict(train_features)), accuracy_score(train_labels, clf1.predict(train_features))

print 'Test Accuracy:', accuracy_score(test_labels, clf.predict(test_features)), accuracy_score(test_labels, clf1.predict(test_features))

#print clf.predict(test_features)
