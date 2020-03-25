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

#saves generated plot in correct output director/sub_directory
def saveFig(outputDirectory, fileNameIn, fileAddendum):
    sub_dir = ''
    #Switch statement to store in proper subdirectory based on fileName
    if 'Cy' in fileNameIn:
        sub_dir = "Cycling\\"
    elif 'Dr' in fileNameIn:
        sub_dir = "Driving\\"
    elif 'Ru' in fileNameIn:
        sub_dir = "Running\\"
    elif 'Sit' in fileNameIn:
        sub_dir = "Sitting\\"
    elif 'St' in fileNameIn:
        sub_dir = "StairDown\\"
    elif 'Su' in fileNameIn:
        sub_dir = "StairUp\\"
    elif 'Sd' in fileNameIn:
        sub_dir = "Standing\\"
    elif 'Wa' in fileNameIn:
        sub_dir = "Walking\\"
    
    plt.savefig(outputDirectory+sub_dir+fileNameIn+fileAddendum)


#plots features per file against time
<<<<<<< HEAD
def plotGraph(dataIn, maxY, fileNameIn, fileAddendum, outputDirectory):
=======
def plotTest(dataIn, sub_dir, fileNameIn):

>>>>>>> f6de5424c8260f99a1b94cc8ca8369f2cae51eef
    fig = plt.figure(figsize=(30, 6))
    #fig.canvas.set_window_title('Reading info from excel file')
    for i in range(1,dataIn.shape[1]):
        plt.plot(dataIn[:,0], dataIn[:,i], label='id %s' %i)
    plt.title(fileNameIn+fileAddendum)
    plt.xlabel('Time')
    plt.ylabel('Feature')
    plt.legend()
    plt.xlim(dataIn[0,0],dataIn[dataIn.shape[0]-1,0])
<<<<<<< HEAD
    plt.ylim(0, maxY)

    saveFig(outputDirectory, fileNameIn, fileAddendum)
=======
    plt.ylim(0,500)
    plt.savefig("C:\\Users\\Stephanos\\Documents\\Dev\ML\\CSCI5957MachineLearningPhase1\\test\\"+sub_dir+fileNameIn+"Limit500")
>>>>>>> f6de5424c8260f99a1b94cc8ca8369f2cae51eef
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
<<<<<<< HEAD
    plt.ylim(0, maxY)
    saveFig(outputDirectory, fileNameIn, fileAddendum)
=======
    plt.xlim(dataIn[0,0],dataIn[dataIn.shape[0]-1,0])
    plt.ylim(0,500)
    plt.savefig("C:\\Users\\Stephanos\\Documents\\Dev\ML\\CSCI5957MachineLearningPhase1\\"+fileNameIn+"Limit500")
>>>>>>> f6de5424c8260f99a1b94cc8ca8369f2cae51eef
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

<<<<<<< HEAD
def extractFileName(stringIn):
    return extractSubstring(rExtractSubstring(stringIn, "\\"), ".")
=======
# =========================  Reading  ============================================
#Reads in features by file
def ReadByFile(directory, sub_dir, rowsToSkipUse, colsToUse):
    dataFiles = files.glob(directory+sub_dir+'*.csv')
    for file in dataFiles:
        if file.find("left") > 0 or file.find("right") > 0:
            data = np.loadtxt(file, delimiter=",", skiprows=1, usecols = colsToUse)
            plotTest(data, sub_dir, ExtractSubstring(rExtractSubstring(file,"\\"),"."))
>>>>>>> f6de5424c8260f99a1b94cc8ca8369f2cae51eef

# ======================  Re-scaling  ==================================

#rescales a single data point
def rescale(x, min, max):
    range = max - min
    return ((x-min)/range)

<<<<<<< HEAD
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
def ReadFilePaths(directory, sub_directories):
    fileNames = []
    dataFiles = []
    for sub_dir in sub_directories:
        dataFiles.extend(files.glob(directory+sub_dir+'*.csv'))
    for file in dataFiles:
        if file.find("left") > 0 or file.find("right") > 0:
            fileNames.append(file)
    return fileNames

if __name__ == '__main__':
    global labels
    labels = [ "Time (Milliseconds)", "Time" ,"P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9"]
    directory = 'C:\\Users\\Stephanos\\Documents\\Dev\ML\\CSCI5957MachineLearningPhase1\\rawData\\'
    outputDirectory = 'C:\\Users\\Stephanos\\Documents\\Dev\ML\\CSCI5957MachineLearningPhase1\\test\\'
   
    #Read paths of all data files, in all sub_directories 
    sub_directories = ['Cycling\\', 'Driving\\', 'Running\\', 'Sitting\\', 'StairDown\\', 'StairUp\\', 'Standing\\', 'Walking\\']
    paths = ReadFilePaths(directory, sub_directories)
    
    data = ReadByFileRate(paths,1,(0, 2, 3, 4, 5, 6, 7, 8, 9, 10), 20)
    plotDirectory(data, 500, paths, "max500", outputDirectory)
    col = 2
    data = ReadByFileRate(paths,1,(0, 2), 40)
    data = rescaleSet(data)
    #plotFeature(data, 500, labels[col], "by Feature", outputDirectory, col) #Works finally - looks awful, will need to pass in selected files
