import numpy as np 
from numpy import asarray 
from numpy import savetxt

import matplotlib as plt 
from glob import glob

import wfdb 
from wfdb import processing 

import os   # Making directories to store each record

import zipfile # Compressing the binary files to save space 
import shutil 

from numpy import array 
from numpy import argmax 
 
import keras 
from keras.utils import to_categorical #one-hot encoding method 

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import sys 

'''
Will make a requirements.txt using pip freeze to get all the necessary packages
for this program.  
'''

#----------------------------GETTING_DATA ------------------------------
def get_records():
    '''
    Three file types (.dat, .hea, .atr)
    We only want the .atr data 
    '''
    paths = glob('data/*.atr') #list of all the data files. 

    # Get rid of the extension and sort in increasing order. 
    paths = [path[:-4] for path in paths]
    paths.sort()
    return paths
  
def qrs_detect(records, max_num=48): #There's 48 total records. If not specified, we'll process every single record 
    '''
    qrs_inds list of ints representing the location of QRS in the signals. E.g. Index 0 in
    qrs_inds represents the first location of the QRS complex in signals. 
    signals is a list of the physical signals represented in floating point. 
    '''
    qrs_inds = []
    signals = []
    for e in records[:max_num]: #First 10 elements. Processing all records would take too long for the purpose of this project. 
        if '-' in e: continue #invalid record        
        sig, fields = wfdb.rdsamp(e, channels=[0]) 
        #converting a array of lists to a numpy array holding just the values 
        signals.append(array([x[0] for x in sig])) 
        #run detection and learning algorithm 
        qrs_ind = processing.xqrs_detect(sig=sig[:,0], fs=fields['fs'])
        qrs_inds.append(qrs_ind) 
    return qrs_inds, signals
#----------------------------GETTING_DATA ------------------------------

def segment_QRS(qrs_inds, signals):
    '''
    Need to create a list of numpy arrays, each representing its own QRS
    complex. We'll use 75% of this array to train and 25% to test for
    accuracy. 
    '''
    prev_ind = 0    # Lower bound on segment 
    end_ind = 0;  # Upper bound on segment 
    last_ind = qrs_inds[-1] # Last index in qrs_inds. Used for edge case 
    segments = []   # List of numpy arrays representing ONE patient's QRS complexes
    '''
    Segment from 0 to halfway between elements of indexes 0 and 1. Use that 'halfway' 
    value to then segment between index 1 and 2 and so on. 
    Edge case: Extracting the last segment. 
    '''
    one_behind = 0
    for ind in qrs_inds[0]: 
        if ind == qrs_inds[0][0]: continue # If 'ind' is the first one in the list, skip -- Fencepost algorithm 
        #Case when we just need to iterate from the last prev_ind to the end of signals
        if ind == qrs_inds[0][-1]: 
            '''
            Special case where we want to just get the prev_ind to the end of the list. Fence post case.
            Take values from prev_ind to len(signals[0] - 1)
            '''
            segments.append(signals[prev_ind:len(signals) - 1])
            continue
        end_ind = ((qrs_inds[0][one_behind] + ind) // 2) - 1
        segments.append(signals[prev_ind:end_ind])
        '''
        prev_ind and end_ind are what splits each QRS complex from the next. 
        Split the numpy array based on this then save as binary files. 
        Example: segments.append(signals[0].split(prev_ind:end_ind)) for every patient. 
        '''
        prev_ind = end_ind + 1
        one_behind = one_behind + 1
    return segments 

#----------------------------One hot encoding----------------------------
def one_hot_encode(qrs_segments, record_name):
    #np.set_printoptions(threshold=sys.maxsize) # For debug purposes. 
    filepath = os.getcwd() + '/encoded_record_segments/'
    #Make directory to hold all the different records. 
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    '''
    sub_filepath is the directory under which all encoded segments for that
    particular record will be stored  
    '''    
    sub_filepath = filepath + record_name 
    if not os.path.exists(sub_filepath): 
        os.makedirs(sub_filepath)
    #If we've already dealt with this specific record. 
    if os.path.exists(sub_filepath + '.zip'):
        print('{0}\'s directory already exists. Skipping this record.'.format(record_name))
        return 
    
    seg_filepath = sub_filepath + '/seg'    # Name of the actual files we'll use. E.g. seg0, seg1, seg2 = the binary files. 
    counter = 0     # For appending a number to make each filename unique. 
    for qrs_complex in qrs_segments:
        #Integer encoding: Need to convert the floating point values to integers for binary encoding. 
        label_encoder = LabelEncoder() 
        integer_encoded = label_encoder.fit_transform(qrs_complex) 
        #Binary encoding:One-hot encoding
        onehot_encoder = OneHotEncoder(sparse=False) 
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1) 
        onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
        np.save(seg_filepath + str(counter), onehot_encoded)    # Saving as a binary file (numpy array) 
        counter = counter + 1 
    #Need to compress folder here. 
    shutil.make_archive(sub_filepath, 'zip', sub_filepath)
    shutil.rmtree(sub_filepath) # Removes the original directory to save space. 


#------------------------------------------------------------------------
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
    '''
    b = np.load(os.getcwd() + '/encoded_record_segments/record100/seg0.npy')
    np.set_printoptions(threshold=sys.maxsize)
    '''
    
    records = get_records() #all of our atr files alphabetized.
    
    '''
    Change the 2nd param in qrs_detect to indicate how many records to run
    the algorithm on. 
    Calling 'qrs_detect(records)' will, by default, run the program on all 48
    records. Having '1' will run it on the first record, '5' will run on the first 5 
    records and so on. 
    '''
    qrs_inds, signals = qrs_detect(records, 11) #
    for qrs, sig, record in zip(qrs_inds, signals, records): 
        segments = segment_QRS(qrs_inds, sig) 
        one_hot_encode(segments, 'record' + record[5:]) # [5:] will get rid of the 'data' prefix in the path. 
    