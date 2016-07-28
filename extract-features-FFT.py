"""Script extracts the frequencies from the dataset and makes frequency "prints" of all .wav files passed as input.

IN: Paths to directories consisting of .wav files.
OUT: Saved .fft.npy files for respective .wav files in input directories.

Run instructions:
python extract-features-FFT.py train_dir_path_1 train_dir_path_2 ... train_dir_path_N

Where train_dir_path_i consists of .wav files.

NOTE:
1. Use ONLY absolute paths. 
"""

import scipy
import scipy.io.wavfile
import os
import sys
import glob
import numpy as np

# Extracts frequencies from a wavile and stores in a file
def create_fft(wavfile): 
	sample_rate, song_array = scipy.io.wavfile.read(wavfile)
	fft_features = abs(scipy.fft(song_array[:10000]))
	base_fn, ext = os.path.splitext(wavfile)
	data_fn = base_fn + ".fft"
	np.save(data_fn, fft_features)


def main():
	train_dirs = sys.argv[1:]
	for train_dir in train_dirs:
		os.chdir(train_dir)
		for wavfile in os.listdir(train_dir):
			# print(wavfile)
			create_fft(wavfile)
		# os.system("rm *.wav")
		# os.system("ls")


if __name__ == "__main__":
	main()