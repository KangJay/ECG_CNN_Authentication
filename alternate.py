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

    


#------------------------------------------------------------------------
def get_signals(records, max_num=48): #There's 48 total records. If not specified, we'll process every single record 
    signals = []
    for e in records[:max_num]: #First 10 elements. Processing all records would take too long for the purpose of this project. 
        if '-' in e: continue #invalid record        
        sig, fields = wfdb.rdsamp(e, channels=[0]) 
        #converting a array of lists to a numpy array holding just the values 
        signals.append(array([x[0] for x in sig])) 
    return  signals
#----------------------------GETTING_DATA ------------------------------
if __name__ == '__main__': 
    #https://blog.goodaudience.com/introduction-to-1d-convolutional-neural-networks-in-keras-for-time-sequences-3a7ff801a2cf
    records = get_records() #all of our atr files alphabetized.
    signals = get_signals(records, 11)
    records = records[:11] #We only want the first 11 records.
    
    filepath = os.getcwd() + '/all_signals/'
    #Make directory to hold all the different records. 
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    '''
    sub_filepath is the directory under which all encoded segments for that
    particular record will be stored  
    '''
    #https://www.freecodecamp.org/news/how-to-combine-multiple-csv-files-with-8-lines-of-code-265183e0854/
    
    INDEX_DIFF = 1249
    for record, signal in zip(records, signals):
        number = 0
        record = record[5:]  
        '''
        Segment each signal into batches of 1250 values. 
        Should give the neural network a sense of temporal distance. 
        There are ALWAYS 650,000 values in each signal. 
        650,000/1250 = 520 which is our range(len(x)) value. 
        Each record length is going to be 521. 521st element is the label. 
        '''         
        index = 0
        l = []
        sub_filepath = filepath + record + ".csv"
        for x in range(520):
            #sub_filepath = filepath + record + '_' + str(number) + ".csv" #Just meant to guarantee a unique name.
            #segmented = signal[index:index + INDEX_DIFF + 1]
            #segmented = np.array(signal[index:index + INDEX_DIFF + 1])
            segmented = None
            if record == '100':
                segmented = np.append(signal[index:index + INDEX_DIFF + 1], [1])            
            else: 
                segmented = np.append(signal[index:index + INDEX_DIFF + 1], [0])
            l.append(segmented)
            #np.savetxt(sub_filepath, [segmented], delimiter=",")
            index += INDEX_DIFF + 1
            number += 1
        np.savetxt(sub_filepath, l, delimiter=",")
