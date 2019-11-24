from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPool2D
from keras.layers import Flatten

import numpy as np 

#Needed for unpacking the files 
import shutil
import os

import model

'''
1. Try to use the base signals (650000 index values).
2. Possibly one-hot encode to 1's and 0's and run the same
3. Use the binary images 
'''


#/encoded_record_segments/
#https://www.tensorflow.org/tutorials/images/cnn
#https://stackoverflow.com/questions/46204569/how-to-handle-variable-sized-input-in-cnn-with-keras
#https://blog.goodaudience.com/introduction-to-1d-convolutional-neural-networks-in-keras-for-time-sequences-3a7ff801a2cf

def extract_records(deleteZips=True):
    '''
    Short-hand function to extract all of our extracted files for training the model. 
    All the binary files are currently zip-archived in order to exponentially save space.
    A typical record's binary files are 300-400 MB each. Compressed, they're 2-3 MB. 
    '''
    all_zip_files = os.listdir(os.getcwd() + '/encoded_record_segments')
    extract_dir = os.getcwd() + '/encoded_record_segments'
    for file in all_zip_files:
        filename = extract_dir + '/' + file
        shutil.unpack_archive(filename, filename[:-4], 'zip')
        if deleteZips is True: 
            os.remove(filename)

if __name__ == '__main__': 
    print(model.model)
