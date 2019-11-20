import numpy as np 
import matplotlib as plt 
from glob import glob
import wfdb 
from numpy import array 
from numpy import argmax 
import tensorflow as tf 
import keras 
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

    # Get rid of the extension
    paths = [path[:-4] for path in paths]
    paths.sort()
    return paths
  
def segment(records, max_num=48): #There's 48 total records. If not specified, we'll process every single record 
    Normal = []
    #data/100
    #signals, field = wfdb.rdsamp('data/100', channels=[0])
    #print(signals, field)
    for e in records[:max_num + 1]: #First 10 elements. Processing all records would take too long for the purpose of this project. 
        if '-' in e: continue #invalid record 

        signals, fields = wfdb.rdsamp(e, channels=[0]) 
        '''
        ann = wfdb.rdann(e, 'atr')
        good = ['N']
        ids = np.in1d(ann.symbol, good)
        imp_beats = ann.sample[ids]
        beats = (ann.sample)
        for i in imp_beats:
            beats = list(beats)
            j = beats.index(i)
            if(j!=0 and j!=(len(beats)-1)):
                x = beats[j-1]
                y = beats[j+1]
                diff1 = abs(x - beats[j])//2
                diff2 = abs(y - beats[j])//2
                Normal.append(signals[beats[j] - diff1: beats[j] + diff2, 0])
        '''
        Normal.append(signals) 
    return Normal
#----------------------------GETTING_DATA ------------------------------

#----------------------------One hot encoding----------------------------
def one_hot_encode(segments):
    pass


#------------------------------------------------------------------------


if __name__ == '__main__':  
    records = get_records() #all of our atr files alphabetized. 
    segments = segment(records, 1) 
    '''
    https://wfdb.readthedocs.io/en/latest/processing.html
    Need to segment the QRS using qrs_detect from wfdb
    '''
    print(segments)
    #print(type(segments[0]))
    #print(len(records))
    #segments = segmentation(records) 
    #one_hot_encode(segments) 
