import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPool2D
from keras.layers import Flatten

import numpy as np 
from numpy import genfromtxt # processing our csv files. 

#Needed for unpacking the files 
import shutil
import os
import tensorflow as tf


'''
1. Try to use the base signals (650000 index values).
2. Possibly one-hot encode to 1's and 0's and run the same
3. Use the binary images 
'''


#/encoded_record_segments/
#https://www.tensorflow.org/tutorials/images/cnn
#https://stackoverflow.com/questions/46204569/how-to-handle-variable-sized-input-in-cnn-with-keras
#https://blog.goodaudience.com/introduction-to-1d-convolutional-neural-networks-in-keras-for-time-sequences-3a7ff801a2cf
'''
def extract_records(deleteZips=True):
    all_zip_files = os.listdir(os.getcwd() + '/encoded_record_segments')
    extract_dir = os.getcwd() + '/encoded_record_segments'
    for file in all_zip_files:
        filename = extract_dir + '/' + file
        shutil.unpack_archive(filename, filename[:-4], 'zip')
        if deleteZips is True: 
            os.remove(filename)
'''


def partition_data():
    '''
    train_data is considered 80% of the data available for training the model. 
    test_data represents 20% and will be used to test the accuracy and loss function
    of the model after training. 
    '''
    test_data, test_labels = [], []
    filepath = os.getcwd() + '/all_signals/'
    data = genfromtxt(filepath + '100.csv', delimiter=',')
    '''
    Each record has exactly 520 records of 1200 elements long + 1 element at the end
    which is our label.  
    80% of 520 records is 416 records. 
    '''

    train_index = int(len(data) * 0.8)
    data_index = len(data[0])

    train_data = [x[0:data_index] for x in data[0:train_index]]
    test_data = [x[0:data_index] for x in data[train_index + 1:]]

    train_labels = []
    test_labels = []

    '''
    Positive data and labels done. Need to partition out our other records. 
    'number' represents the file number. Record 10 is ommited from the data file from MIT BIH
    So we skip over record 10. 
    '''

    '''
    print(len(train_data))
    print(len(train_labels))
    print(len(test_data))
    print(len(test_labels))
    '''

    record_names = [str(x) + '.csv' for x in range(101, 112) if x != 110] 
    for record in record_names:
        data = genfromtxt(filepath + record, delimiter=',')
        #print(len(data))
        #negative will follow as: 160 records from each record for training and 40 for test.
        neg_train_data = [x[0:data_index] for x in data[0:160]]
        neg_test_data = [x[0:data_index] for x in data[161:201]]
        
        for test_d in neg_test_data:
            test_data.append(test_d)

        for train_d in neg_train_data:
            train_data.append(train_d)
    
    np.random.shuffle(train_data)
    np.random.shuffle(test_data) 
    train_labels = np.array([int(x[-1]) for x in train_data])
    test_labels = np.array([int(x[-1]) for x in test_data])
    train_data = np.array([x[0:data_index - 1] for x in train_data])
    test_data = np.array([x[0:data_index - 1] for x in test_data])

    train_labels = keras.utils.to_categorical(train_labels)
    test_labels = keras.utils.to_categorical(test_labels)

    train_data = train_data.reshape((2016, 1250, 1))
    test_data = test_data.reshape((503, 1250, 1))
    #Save files as binary files to save time on training. 
    np.save(filepath + 'traindata', train_data)
    np.save(filepath + 'trainlabels', train_labels)
    np.save(filepath + 'testdata', test_data)
    np.save(filepath + 'testlabels', test_labels)


######################################################################################
if __name__ == '__main__': 
    #partition_data()
    from cnn_model import model
    filepath = os.getcwd() + '/all_signals/'  
    train_data = np.load(filepath + 'traindata.npy')
    train_labels = np.load(filepath + 'trainlabels.npy')
    test_data = np.load(filepath + 'testdata.npy')
    test_labels = np.load(filepath + 'testlabels.npy')
    #train_labels = keras.utils.to_categorical(train_labels)
    #test_data = keras.utils.to_categorical(test_labels)
    #test_data = np.expand_dims(test_data, axis=2)
    #print(model.summary())
    #print(train_data)
    #Hyper parameter. Fine-tune then run again
    BATCH_SIZE = 32  
    EPOCHS = 10

    callbacks_list = [
        keras.callbacks.ModelCheckpoint(
            filepath=os.getcwd(), 
            monitor='val_loss',
            save_best_only=True),
        keras.callbacks.EarlyStopping(monitor='acc', patience=1)
    ]
    #print(model.summary())
    model.compile(loss='categorical_crossentropy',
        optimizer='adam', metrics=['accuracy'])

    history = model.fit(
        train_data, train_labels,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks_list,
        #validation_data=(test_data, test_labels)
    )
    score = model.evaluate(test_data, test_labels, verbose=0)
    print('Test loss: {}'.format(score[0]))
    print('Test accuracy: {}'.format(score[1]))
    
    model.save(filepath + 'modelv1')

###########################################################################################    
