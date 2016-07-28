"""Script reads from fft and mfcc files and trains using logistic regression and knn

IN: Paths to directories consisting of FFT files, and MFCC files.
OUT: Splits dataset as per code into train and test sets, performs training and tests. Displays classification accuracy along with confusion matrix.

Run instructions:
python train-classify.py path_dir1 path_dir2 

Where path_dir1 is the base_dir that consists of subdirs consisting of fft files.
	  path_dir2 is the base-dir that consists of subdirs consisting of mfcc files.

NOTE:
1. Use ONLY absolute paths. 
"""

import sklearn 
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import scipy
import os
import sys
import glob
import numpy as np

"""reads FFT-files and prepares X_train and y_train.
genre_list must consist of names of folders/genres consisting of the required FFT-files
base_dir must contain genre_list of directories
"""
def read_fft(genre_list, base_dir):
	X = []
	y = []
	for label, genre in enumerate(genre_list):
		# create UNIX pathnames to id FFT-files.
		genre_dir = os.path.join(base_dir, genre, "*.fft.npy")
		# get path names that math genre-dir
		file_list = glob.glob(genre_dir)
		for file in file_list:
			fft_features = np.load(file)
			X.append(fft_features)
			y.append(label)
	
	# print(X)
	# print(y)
	# print(len(X))
	# print(len(y))

	return np.array(X), np.array(y)


"""reads MFCC-files and prepares X_train and y_train.
genre_list must consist of names of folders/genres consisting of the required MFCC-files
base_dir must contain genre_list of directories
"""
def read_ceps(genre_list, base_dir):
	X, y = [], []
	for label, genre in enumerate(genre_list):
		for fn in glob.glob(os.path.join(base_dir, genre, "*.ceps.npy")):
			ceps = np.load(fn)
			num_ceps = len(ceps)
			X.append(np.mean(ceps[int(num_ceps*1/10):int(num_ceps*9/10)], axis=0))
			y.append(label)
	
	return np.array(X), np.array(y)


def learn_and_classify(X_train, y_train, X_test, y_test, genre_list):

	# print("X_train = " + str(len(X_train)), "y_train = " + str(len(y_train)), "X_test = " + str(len(X_test)), "y_test = " + str(len(y_test)))
	logistic_classifier = linear_model.LogisticRegression()
	logistic_classifier.fit(X_train, y_train)
	logistic_predictions = logistic_classifier.predict(X_test)
	logistic_accuracy = accuracy_score(y_test, logistic_predictions)
	logistic_cm = confusion_matrix(y_test, logistic_predictions)
	print("logistic accuracy = " + str(logistic_accuracy))
	print("logistic_cm:")
	print(logistic_cm)

	# knn_classifier = KNeighborsClassifier()
	# knn_classifier.fit(X_train, y_train)
	# knn_predictions = knn_classifier.predict(X_test)
	# knn_accuracy = accuracy_score(y_test, knn_predictions)
	# knn_cm = confusion_matrix(y_test, knn_predictions)
	# print("knn accuracy = " + str(knn_accuracy))
	# print("knn_cm:") 
	# print(knn_cm)
	
	
	plot_confusion_matrix(logistic_cm, "Confusion matrix", genre_list)
	# plot_confusion_matrix(knn_cm, "Confusion matrix for FFT classification", genre_list)


def plot_confusion_matrix(cm, title, genre_list, cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(genre_list))
    plt.xticks(tick_marks, genre_list, rotation=45)
    plt.yticks(tick_marks, genre_list)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def main():
	# first command line argument is the base folder that consists of the fft files for each genre
	base_dir_fft  = sys.argv[1]
	# second command line argument is the base folder that consists of the mfcc files for each genre
	base_dir_mfcc = sys.argv[2]
	
	"""list of genres (these must be folder names consisting .wav of respective genre in the base_dir)
	Change list if needed.
	"""
	genre_list = ["classical", "jazz", "country", "pop", "rock", "metal"]
	
	#genre_list = ["classical", "jazz"] IF YOU WANT TO CLASSIFY ONLY CLASSICAL AND JAZZ

	# use FFT
	X, y = read_fft(genre_list, base_dir_fft)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .25)

	# print("X_train = " + str(len(X_train)), "y_train = " + str(len(y_train)), "X_test = " + str(len(X_test)), "y_test = " + str(len(y_test)))
	
	print('\n******USING FFT******')
	learn_and_classify(X_train, y_train, X_test, y_test, genre_list)
	print('*********************\n')

	# use MFCC
	X, y = read_ceps(genre_list, base_dir_mfcc)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
	print('******USING MFCC******')
	learn_and_classify(X_train, y_train, X_test, y_test, genre_list)
	print('*********************')
	

if __name__ == "__main__":
	main()