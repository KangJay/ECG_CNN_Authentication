import numpy as np 
import matplotlib as plt 
from glob import glob

import wfdb 
from wfdb import processing 

from numpy import array 
from numpy import argmax 
import tensorflow as tf 
import keras 

import sys 
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
  
def qrs_segment(records, max_num=48): #There's 48 total records. If not specified, we'll process every single record 
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
        signals.append(sig) 
        #run detection and learning algorithm 
        qrs_ind = processing.xqrs_detect(sig=sig[:,0], fs=fields['fs'])
        qrs_inds.append(qrs_ind) 
    return qrs_inds, signals
#----------------------------GETTING_DATA ------------------------------

#----------------------------One hot encoding----------------------------
def one_hot_encode(segments):
    pass


#------------------------------------------------------------------------


if __name__ == '__main__':  
    records = get_records() #all of our atr files alphabetized. 
    qrs_inds, sig = qrs_segment(records, 1) 
    np.set_printoptions(threshold=sys.maxsize)
    '''
    For signals, it is a numpy array of a bunch of lists with only one element
    inside. I'm not sure why wfdb formats it this way but I could tweak it later on
    for enhancement of usability.  
    '''
    #print(qrs_inds[0][0])
    #print(sig[0][qrs_inds[0][0]])
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
    print(sig[0][qrs_inds[0][0] - 2], sig[0][qrs_inds[0][0] - 1], sig[0][qrs_inds[0][0]], sig[0][qrs_inds[0][0] + 1], sig[0][qrs_inds[0][0] + 2])
    print(sig[0][qrs_inds[0][1] - 2], sig[0][qrs_inds[0][1] - 1], sig[0][qrs_inds[0][1]], sig[0][qrs_inds[0][1] + 1], sig[0][qrs_inds[0][1] + 2])
    print(sig[0][qrs_inds[0][2] - 2], sig[0][qrs_inds[0][2] - 1], sig[0][qrs_inds[0][2]], sig[0][qrs_inds[0][2] + 1], sig[0][qrs_inds[0][2] + 2])