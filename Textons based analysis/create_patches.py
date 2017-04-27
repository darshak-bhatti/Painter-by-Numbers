################################################################################################################################
#This file was created by Mihir Mirajkar.
#Unity ID: mmmirajk
################################################################################################################################

from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import glob
from multiprocessing import Pool
from PIL import Image, ImageFilter
from sklearn import cluster
import pickle
import random
from scipy.misc import toimage

random.seed(1)

def labelled_data_dict(image_file):
	f = open(image_file, 'r')
	paintings_dict = {}
	for i,line in enumerate(f):
		painting, artist = line[:-1].split(',')[:2]
		try:
			paintings_dict[artist].append(painting)
		except Exception:
			paintings_dict[artist] = [painting]
	return paintings_dict

paintings_artist_dict = labelled_data_dict('train_info.csv')
print len(paintings_artist_dict)
for e in paintings_artist_dict:
	if len(paintings_artist_dict[e]) > 120:
		selected = e
		to_process = paintings_artist_dict[e]
		break

i = 0
extras = []
while i < 65:
	x = random.randint(0,len(paintings_artist_dict.items()))
	if paintings_artist_dict.items()[x][0] == selected:
		continue
	y = random.randint(0, len(paintings_artist_dict.items()[x][1]))
	extras.append(paintings_artist_dict.items()[x][1][0])
	i += 1

print 'i:', i
to_process += extras
print to_process
print len(to_process)

def patches(file):
	file = 'train/'+file

	jpgfile = Image.open(file).convert('L')
	print file
	t = np.asarray(jpgfile)
	print t.shape
	if t.shape[0] > 512 or t.shape[1] > 512:
		for i in range(1500):
			x = random.randint(0,t.shape[0] - 35)
			y = random.randint(0,t.shape[1] - 35)		
			temp = np.array(t[x:x+35, y:y+35])
			toimage(temp).save('patches2/'+file[6:]+'_patch'+str(i)+'.jpg')

def main():
	new_ds = []
	thread_count = 50

	pool = Pool(thread_count)
	pool.map(patches, to_process)
	pool.close()  	
	pool.join() 

main()
