from numpy import genfromtxt
import os

data = genfromtxt(os.getcwd() + '/all_signals/100.csv', delimiter=',')
print(data)
