import pandas as pd
from pandas import read_table 
import numpy as np
import datetime as dt
from DataUtil import DataUtil as Data
from FileReaderUtil import FileReader
from GraphUtil import GraphUtil as Graph
from SynchronizationUtil import SynchronizationUtil as Synchronization
from FeatureUtil import FeatureUtil as Feature
try:
    # [OPTIONAL] Seaborn makes plots nicer
    import seaborn
except ImportError:
    pass

    #Wrapper function that takes file paths, the data read rate and builds raw data plots, saving them in the provided outputDirectory
    def graphRawData(paths, rate, outputDirectory):
        grapher = Graph()
        activityData = FileReader.ReadByFileRate(paths,1,(0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), rate)
        for activity in activityData:
            grapher.plotDirectory(activityData[activity], 500, paths[activity], "max500", outputDirectory)

    #Wrapper function that takes paths of raw data and returns a dictionary of activities containing lists of events containing a list of dataframes
    #each dataframe represents a two second chunk from the event. Trimming and combining also occur in this proccess
    def synchronizeDataFromPaths(paths):
        activityDFs = {'Cycling': [], 'Driving': [], 'Running': [], 'Sitting': [], 'StairDown': [], 'StairUp': [], 'Standing': []}
        activityData = FileReader.ReadByFileEvent(paths,1,(0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12))
        tempEvents = []
    
        for activity in activityData:
            for event in activityData[activity]:
                tempEvent = Data.shoeDataToDataFrame(event)
                tempEvent = Synchronization.trim_start_end_times(tempEvent)
                tempEvent = Synchronization.join_event_data(tempEvent)
                activityDFs[activity].append(Synchronization.chunkify_data_frame(tempEvent))

        return activityDFs

if __name__ == '__main__':
    # Set up enviornemnt constants and read in file paths
    directory = 'C:\\Users\\Stephanos\\Documents\\Dev\\ML\\CSCI5957MachineLearningPhase1\\rawData\\'
    outputDirectory = 'C:\\Users\\Stephanos\\Documents\\Dev\ML\\CSCI5957MachineLearningPhase1\\test\\'
    sub_directories = ['Cycling', 'Driving', 'Running', 'Sitting', 'StairDown', 'StairUp', 'Standing']
    paths = FileReader.ReadFilePaths(directory, sub_directories)

    # -- Graph all the raw data --
    #graphRawData(paths, 40, outputDirectory)

    # -- Synchronizing data  -- NEED to be split up by type first!
    activityDFs = synchronizeDataFromPaths(paths)

    # -- Testing Feature methods --  
    #   print("Min: "+Feature.findMin(newData))
    
    #print("Max: ")
    #print(Feature.findMax(newData))
    #print("Min: ")
    #print(Feature.findMin(newData))
    #print("Median: ")
    #print(Feature.findMedian(newData))
    #print("Mode: ")
    #print(Feature.findMode(newData))
    #print("Sum: ")
    #print(Feature.findSum(newData))
    #print("StDev: ")
    #print(Feature.findStdDev(newData))
    #print("Kurtosis: ")
    #print(Feature.findKurtosis(newData))
    #print("Area Under Curve: ")
    #print(Feature.findAreaUnderCurve(newData))
    #print("Avg Slope: ")
    #print(Feature.findAvgSlope(newData))
    #print("Skewness: ")
    #print(Feature.findSkewness(newData))
    #print("Time to peak: ")
    #print(Feature.findTimeToPeak(newData))
    

    # -- Plotting features --
    #col = 2
    #data = FileReader.ReadByFileRate(paths,1,(0, 2), 40)
    #data = Data.rescale2D(data)
    #grapher.plotFeature(data, 500, labels[col], "by Feature", outputDirectory, col) #Works finally - looks awful, will need to pass in selected files