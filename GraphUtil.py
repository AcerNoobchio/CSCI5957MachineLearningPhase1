import matplotlib.pyplot as plt
import numpy as np
from StringUtil import StringUtil as String


class GraphUtil:
    """Contains utility methods used for plotting graphs"""
    def __init__(self):
        self.labels = [ "Time (Milliseconds)", "Time" ,"P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9"]

    
    #saves generated plot in correct output director/sub_directory
    def saveFig(self, outputDirectory, fileNameIn, fileAddendum):
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

    def plotArray(self, dataIn, maxY, minX, xLabel, yLabel, fileNameIn, fileAddendum, outputDirectory):
        fig = plt.figure(figsize=(30, 6))
        
        plt.plot(dataIn, label='id %s' %fileNameIn, marker='o')
        plt.title(fileNameIn+fileAddendum)
        plt.xlabel('Time')
        plt.ylabel('Feature')
        plt.legend()
        plt.ylim(0, maxY)
        plt.xlim(minX, len(dataIn)-1)
        plt.savefig(outputDirectory+fileNameIn+fileAddendum)
        plt.close()
    #plots features per file against time
    def plotGraph(self, dataIn, maxY, fileNameIn, fileAddendum, outputDirectory):
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

        self.saveFig(outputDirectory, fileNameIn, fileAddendum)
        #plt.show()

        plt.close()

    #Need to edit to make the labels current, otherwise will plot data in a 2d array
    def plotFeature(self, dataIn, maxY, fileNameIn, fileAddendum, outputDirectory, col):

        fig = plt.figure(figsize=(30, 20))
        for line in dataIn:
            plt.plot(line[:,0], line[:,1], label='id %s' %fileNameIn)
        plt.title(fileNameIn+" Limit: 500")
        plt.xlabel('Time')
        plt.ylabel(self.labels[col])
        plt.legend()
        plt.ylim(0, maxY)
        self.saveFig(outputDirectory, fileNameIn, fileAddendum)
        #plt.show()

        plt.close()

    def plotDirectory(self, dataIn, maxY, filePaths, plotDescription, outputDirectory):
        for i in range(0,len(dataIn)):
            GraphUtil.plotGraph(self, dataIn[i], maxY, String.extractFileName(filePaths[i]), plotDescription, outputDirectory)

    def plotRankedFeatures(dataIn, filenameIn, outputDirectory):
        dataIn.plot(kind='bar', x='Feature', y='Normalized Variance')
        
        plt.savefig(outputDirectory+sub_dir+fileNameIn+fileAddendum)

