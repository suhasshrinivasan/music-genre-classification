"""Converts .au to .wav file using the sox tool.

IN: Paths to directory consisting of .au files.
OUT: Complete conversion of files in respective directories passed as inputs.

Run instructions:
python convert-to-wav.py path_dir_1 path_dir_2 ... path_dir_N

Where path_dir_i consists of .au files to be converted

NOTE: 
1. .au files will be DELETED. Make sure you have a backup of it to be safe.
2. Use ONLY absolute paths.
3. Run as per instruction or will lead to disastrous results
"""

import sys
import os

# Store all command line args in genre_dirs
genre_dirs = sys.argv[1:]

for genre_dir in genre_dirs:
	# change directory to genre_dir
	os.chdir(genre_dir)

	# echo contents before altering
	print('Contents of ' + genre_dir + ' before conversion: ')
	os.system("ls")

	# loop through each file in current dir
	for file in os.listdir(genre_dir):
		# SOX
		os.system("sox " + str(file) + " " + str(file[:-3]) + ".wav")
	
	# delete .au from current dir
	os.system("rm *.au")
	# echo contents of current dir
	print('After conversion:')
	os.system("ls")
	print('\n')

print("Conversion complete. Check respective directories.")
