from keras.layers import Dense
from keras.models import Sequential

from keras.layers import Conv1D
from keras.layers import MaxPool1D
from keras.layers import GlobalAveragePooling1D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Reshape

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

INPUT_SIZE = 1250 #Number of features in 1 timestep
NUM_CLASSES = 2 #Either is valid authentication or isn't. 
DROPOUT_VALUE = 0.5

model = Sequential()
#model.add(Dense(units=32, input_shape=(1250, 1)))
#model.add(Dense(units=32, input_shape=(INPUT_SIZE, 1))) # Fine-tune hyperparameter
model.add(Conv1D(100, 10, activation='relu', input_shape=(1250, 1)))
model.add(Conv1D(100, 10, activation='relu'))

'''
Limit output vector to be one-third the size of input
'''
model.add(MaxPool1D(3)) 
model.add(Conv1D(160, 10, activation='relu'))
model.add(Conv1D(160, 10, activation='relu'))
'''
Taking the average weights in the neural network
''' 
model.add(GlobalAveragePooling1D())
'''
Assigning 0 weights to DROPOUT_VALUE * 100 percent of neurons
will allow the model to be less sensitive to small variations
in the data since ECG data is 
'''
model.add(Dropout(DROPOUT_VALUE))
'''
Compress the x-dimension vector to NUM_CLASSES. 
The dense layer represents the probability of being one class or 
another. In the case for authentication, valid authentication
& invalid.
'''
model.add(Dense(units=NUM_CLASSES, activation='softmax'))