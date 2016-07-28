# Welcome to the music-genre-classification wiki!

**Language used for development**: Python 2.7.6


This repository consists of development code that classifies music according to the following genres: 

1. Classical (Western)

2. Country (Western)

3. Jazz

4. Pop

5. Rock

6. Metal


**Dataset used for training**: http://opihi.cs.uvic.ca/sound/genres.tar.gz. 

### Features used: 

1. Frequencies (Classification accuracy: ~45%)

2. MFCC (Classification accuracy: ~70%)

### Choice of classifier:

Logistic Regression classifier.


### Consists of Python scripts that do the following:

1. Batch conversion of .au to .wav files.

2. Plotting spectrogram of a .wav file.

3. Extraction of frequencies from the dataset and making frequency "prints" of all .wav files. 

4. Extraction of MFCC from the dataset and making MFCC "prints" of all .wav files.

5. Reading of FFT and MFCC prints, separately, and training suitable classifiers. 
(Refer to individual files for detailed instructions)


## How to use project for testing:

1. Download dataset from: http://opihi.cs.uvic.ca/sound/genres.tar.gz.

2. Extract into suitable directory: BASE_DIR

3. Run convert-to-wav.py on each subdir of BASE_DIR.

4. Run extract-features-FFT.py on each subdir of BASE_DIR.

5. Run extract-features-MFCC.py on each subdir of BASE_DIR.

6. Run train-classify.py according to run instruction provided in the code file.


Reference guide: Building Machine Learning Systems with Python. ISBN 978-1-78216-140-0.
