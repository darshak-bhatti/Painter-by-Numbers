'''
THIS CODE WRITTEN BY ERICK DRAAYER
ID: ecdraaye

Our KNN implementations
'''



import numpy as np
import argparse
import imutils
import cv2
import os


from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from imutils import paths

# Implementation 1, converts images to 64x64, looks at raw pixel intensities 
def image_to_feature_vector(image, size=(64,64)):
	return cv2.resize(image, size).flatten()

# Implementation 2, creates histogram of colors
def extract_color_histogram(image, bins=(8, 8, 8)):
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
		[0, 180, 0, 256, 0, 256])
 
	if imutils.is_cv2():
		hist = cv2.normalize(hist)
 
	else:
		cv2.normalize(hist, hist)
 
	return hist.flatten()

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-k", "--neighbors", type=int, default=1,
	help="# of nearest neighbors for classification")
ap.add_argument("-j", "--jobs", type=int, default=-1,
	help="# of jobs for k-NN distance (-1 uses all available cores)")
args = vars(ap.parse_args())

print("describing images")
imagePaths = list(paths.list_images(args["dataset"]))
 
RawData = []
f = []  # features
l = []  # labels


for (i, imagePath) in enumerate(imagePaths):
	image = cv2.imread(imagePath)
	if image is None:
		continue
	label = imagePath.split(os.path.sep)[-1].split(".")[0]

	pixels = image_to_feature_vector(image)

	hist = extract_color_histogram(image)

	RawData.append(pixels)
	f.append(hist)
	l.append(label)
 
	if i > 0 and i % 1000 == 0:
		print("processed {}/{}".format(i, len(imagePaths)))

RawData = np.array(RawData)
f = np.array(f)
l = np.array(l)
print("pixels matrix: {:.2f}MB".format(RawData.nbytes / (1024 * 1000.0)))
print("f matrix: {:.2f}MB".format(f.nbytes / (1024 * 1000.0)))

(trainRI, testRI, trainRL, testRL) = train_test_split(RawData, l, test_size=0.25, random_state=42)
(trainFeat, testFeat, trainl, testl) = train_test_split(f, l, test_size=0.25, random_state=42)

print("evaluating raw pixel accuracy")
model = KNeighborsClassifier(n_neighbors=args["neighbors"],n_jobs=args["jobs"])
model.fit(trainRI, trainRL)
acc = model.score(testRI, testRL)
print("raw pixel accuracy is {:.2f}%".format(acc * 100))

print("evaluating histogram accuracy")
model = KNeighborsClassifier(n_neighbors=args["neighbors"], n_jobs=args["jobs"])
model.fit(trainFeat, trainl)
acc = model.score(testFeat, testl)
print("histogram accuracy is {:.2f}%".format(acc * 100))

