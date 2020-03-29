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
        colsPerDataType = {'Shoe': (0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), 'Phone': (1, 4, 6, 8, 10, 12, 14)}
        grapher = Graph()

        for dataType in paths:
            activityData = FileReader.ReadByFileRate(paths[dataType],1,colsPerDataType[dataType],rate)
            for activity in activityData:
                grapher.plotDirectory(activityData[activity], 500, paths[dataType][activity], "max500", outputDirectory)

    #Wrapper function that takes paths of raw data and returns a dictionary of activities containing lists of events containing a list of dataframes
    #each dataframe represents a two second chunk from the event. Trimming and combining also occur in this proccess
    def synchronizeDataFromPaths(paths):
        allDataDFs = {'Shoe': {'Cycling': [], 'Driving': [], 'Running': [], 'Sitting': [], 'StairDown': [], 'StairUp': [], 'Standing': []}, 'Phone': {'Cycling': [], 'Driving': [], 'Running': [], 'Sitting': [], 'StairDown': [], 'StairUp': [], 'Standing': []}}
        activityDFs = {'Cycling': [], 'Driving': [], 'Running': [], 'Sitting': [], 'StairDown': [], 'StairUp': [], 'Standing': []}
        colsPerDataType = {'Shoe': (0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), 'Phone': (1, 4, 6, 8, 10, 12, 14)}

        for dataType in paths:
            activityData = FileReader.ReadByFileEvent(paths[dataType],1,colsPerDataType[dataType])
            tempEvents = []
    
            for activity in activityData:
                for event in activityData[activity]:
                    if dataType == 'Shoe':
                        tempEvents = Data.shoeDataToDataFrame(event)
                    elif dataType == 'Phone':
                        tempEvents = Data.phoneDataToDataFrame(event)
                    else:
                        print('Invalid dataType')
                
                    allDataDFs[dataType][activity].append(tempEvents)
        
        chunkedEvents = []
        for dataType in paths:
            for activity in allDataDFs[dataType]:
                for event in allDataDFs[dataType][activity]:
                    event = Synchronization.trim_start_end_times(event)
                    for index in range(0,len(event)):
                        event[index] = Synchronization.chunkify_data_frame(event[index])
                    chunkedEvents.append(event)
                allDataDFs[dataType][activity] = chunkedEvents
                chunkedEvents = []

        return allDataDFs

if __name__ == '__main__':
    # Set up enviornemnt constants and read in file paths
    directory = 'C:\\Users\\Stephanos\\Documents\\Dev\\ML\\CSCI5957MachineLearningPhase1\\rawData\\'
    outputDirectory = 'C:\\Users\\Stephanos\\Documents\\Dev\ML\\CSCI5957MachineLearningPhase1\\test\\'
    sub_directories = ['Cycling', 'Driving', 'Running', 'Sitting', 'StairDown', 'StairUp', 'Standing']
    paths = FileReader.ReadFilePaths(directory, sub_directories)
    # -- Graph all the raw data --
    #graphRawData(paths, 40, outputDirectory)

    # -- Synchronizing data  -- NEED to be split up by type first!
    allDataDFs = synchronizeDataFromPaths(paths)
    print("wazzzzup")
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