import pandas as pd
import matplotlib.pyplot as plt

def plot():
    dataset = pd.read_csv("data.csv") #Read data from CSV datafile
    plt.title("Heart Rate Signal") #The title of our plot
    plt.plot(dataset.hart) #Draw the plot object
    plt.show() #Display the plot

if __name__ == '__main__': 
    pass