'''
This script performs the basic process for applying a machine learning
algorithm to a dataset using Python libraries.

The four steps are:
   1. Download a dataset (using pandas)
   2. Process the numeric data (using numpy)
   3. Train and evaluate learners (using scikit-learn)
   4. Plot and compare results (using matplotlib)


The data is downloaded from URL, which is defined below. As is normal
for machine learning problems, the nature of the source data affects
the entire solution. When you change URL to refer to your own data, you
will need to review the data processing steps to ensure they remain
correct.

============
Example Data
============
The example is from http://mlr.cs.umass.edu/ml/datasets/Spambase
It contains pre-processed metrics, such as the frequency of certain
words and letters, from a collection of emails. A classification for
each one indicating 'spam' or 'not spam' is in the final column.
See the linked page for full details of the data set.

This script uses three classifiers to predict the class of an email
based on the metrics. These are not representative of modern spam
detection systems.
'''

# Remember to update the script for the new data when you change this URL
#URL = "http://mlr.cs.umass.edu/ml/machine-learning-databases/spambase/spambase.data"

# Uncomment this call when using matplotlib to generate images
# rather than displaying interactive UI.
#import matplotlib
#matplotlib.use('Agg')
import pandas as pd
from pandas import read_table 
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import glob as files
try:
    # [OPTIONAL] Seaborn makes plots nicer
    import seaborn
except ImportError:
    pass

#plots features per file against time
def plotGraph(dataIn, maxY, fileNameIn, fileAddendum, outputDirectory):

    fig = plt.figure(figsize=(30, 6))
    #fig.canvas.set_window_title('Reading info from excel file')
    for i in range(1,dataIn.shape[1]):
        plt.plot(dataIn[:,0], dataIn[:,i], label='id %s' %i)
    plt.title(fileNameIn+fileAddendum)
    plt.xlabel('Time')
    plt.ylabel('Feature')
    plt.legend()
    plt.xlim(dataIn[0,0],dataIn[dataIn.shape[0]-1,0])
    plt.ylim(0, maxY)
    plt.savefig(outputDirectory+fileNameIn+fileAddendum)
    #plt.show()

    plt.close()

#Need to edit to make the labels current, otherwise will plot data in a 2d array
def plotFeature(dataIn, maxY, fileNameIn, fileAddendum, outputDirectory, col):

    fig = plt.figure(figsize=(30, 20))
    for line in dataIn:
        plt.plot(line[:,0], line[:,1], label='id %s' %fileNameIn)
    plt.title(fileNameIn+" Limit: 500")
    plt.xlabel('Time')
    plt.ylabel(labels[col])
    plt.legend()
    plt.ylim(0, maxY)
    plt.savefig(outputDirectory+fileNameIn+fileAddendum)
    #plt.show()

    plt.close()

def plotDirectory(dataIn, maxY, filePaths, plotDescription, outputDirectory):
    for i in range(0,len(dataIn)):
        plotGraph(dataIn[i], maxY, extractFileName(filePaths[i]), plotDescription, outputDirectory)


# ======================String Stuff==================================

def rExtractSubstring(stringIn, Delimiter):
    lastIndex = stringIn.rfind(Delimiter)
    lastIndex += len(Delimiter)
    newString = stringIn[lastIndex:]
    return newString

def extractSubstring(stringIn, Delimiter):
    lastIndex = stringIn.rfind(Delimiter)
    newString = stringIn[:lastIndex]
    return newString

def extractFileName(stringIn):
    return extractSubstring(rExtractSubstring(stringIn, "\\"), ".")

# ======================  Re-scaling  ==================================

#rescales a single data point
def rescale(x, min, max):
    range = max - min
    return ((x-min)/range)

#used for rescaling a single numpy array
def rescaleX(dataIn):
    max = dataIn[dataIn.shape[0]-1][0]
    min = dataIn[0][0]
    for i in range(0, len(dataIn)):
       dataIn[i,0] = rescale(dataIn[i][0], min, max)   
    return dataIn

#Used for rescaling an array of numpy arrays
def rescaleSet(dataIn):
    for i in range(0, len(dataIn)):
        data[i] = rescaleX(dataIn[i])
    return dataIn

# =========================  Reading  ============================================
#Reads in features by file - Currently not working - need to refactor to return data instead of graphing it
def ReadByFile(filePaths, rowsToSkipUse, colsToUse):
    for file in filePaths:
            data = np.loadtxt(file, delimiter=",", skiprows=1, usecols = colsToUse)
            plotGraph(data, 500, extractFileName(file))

def ReadByFileRate(filePaths, rowsToSkip, colsToUse, rate):
    data = []
    for file in filePaths:
            data.append(ReadFile(file, rowsToSkip, colsToUse, rate))
    finalData = np.asarray(data)
    return data

#Reads in the lines of a file, only reading in the only nth file where n is the passed-in rate
def ReadFile(filename, skipRows, colsToUse, rate):
    i = 0
    data = []
    with open(filename) as f:
        for line in f:
            if i % rate == 0:
                data.append(line)
            i+=1
        finalData = np.genfromtxt(data, delimiter=",", skip_header=skipRows, usecols = colsToUse)
        return finalData

#collects all of the filenames in a given directory
def ReadFilePaths(directory):
    fileNames = []
    dataFiles = files.glob(directory+'*.csv')
    for file in dataFiles:
        if file.find("left") > 0 or file.find("right") > 0:
            fileNames.append(file)
    return fileNames

if __name__ == '__main__':
    global labels
    labels = [ "Time (Milliseconds)", "Time" ,"P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9"]
    directory = 'C:\\Users\\jacob\\Desktop\\MachineLearning\\rawData\\'
    outputDirectory = 'C:\\Users\\jacob\\Desktop\\MachineLearning\\test\\'
    paths = ReadFilePaths(directory)
    #data = ReadByFileRate(paths,1,(0, 2, 3, 4, 5, 6, 7, 8, 9, 10), 20)
    #plotDirectory(data, 500, paths, "max500", outputDirectory)
    col = 2
    data = ReadByFileRate(paths,1,(0, 2), 40)
    data = rescaleSet(data)
    plotFeature(data, 500, labels[col], "by Feature", outputDirectory, col) #Works finally - looks awful, will need to pass in selected files