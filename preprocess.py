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
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
'''

'''
Don't really need all of these imports. Will clean up in the next commit. 
Requirements: Need numpy, pandas, wfdb for denoising and segmenting data 
Need Tensorflow and keras package for one-hot-encoding the data. 
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
        if ind == qrs_inds[0][0]: continue 
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
    if os.path.exists(sub_filepath + '.zip'):
        print('{0}\'s directory already exists. Skipping this record.'.format(record_name))
        return 
    
    #np.save(sub_filepath + '/seg1', sub_filepath)
    seg_filepath = sub_filepath + '/seg'    # Name of the actual files we'll use. 
    counter = 0 
    for qrs_complex in qrs_segments:
        #Integer encoding 
        label_encoder = LabelEncoder() 
        integer_encoded = label_encoder.fit_transform(qrs_complex) 
        #Binary encoding:One-hot encoding
        onehot_encoder = OneHotEncoder(sparse=False) 
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1) 
        onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
        np.save(seg_filepath + str(counter), onehot_encoded) 
        counter = counter + 1 
    #Need to compress folder here. 
    shutil.make_archive(sub_filepath, 'zip', sub_filepath)
    shutil.rmtree(sub_filepath) # Removes the original directory to save space. 


#------------------------------------------------------------------------
def extract_records(extract_num=48, deleteZips=True):
    #['record101.zip', 'record104.zip', 'record100.zip', 'record102.zip', 'record103.zip']
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
    
    qrs_inds, signals = qrs_detect(records, 11) 
    for qrs, sig, record in zip(qrs_inds, signals, records): 
        segments = segment_QRS(qrs_inds, sig) 
        one_hot_encode(segments, 'record' + record[5:]) 
    
    '''
    FOR RECORD 100
    Printed out a couple QRS complex values to see the actual curve. 
    Confirmed with the record100signals.txt 
    The average value for a still wave (flat) is around -0.250 for this frequency sampling. 
    The start of the complex value (Q) dips to about -0.400 then increases dramatically
    for the R peak, then decreases back down to around -0.400 then stabilizes. 
    QRS complex detection complete. Will need to segment, one-hot encode then start training
    the model using TensorFlow. 
    These values were printed just to see that it is indeed the R peak. 
    '''
    