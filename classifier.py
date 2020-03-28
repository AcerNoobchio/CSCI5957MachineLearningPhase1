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

#Wrapper function that takes directory, subdirectories and rate and builds raw data plots
    def graphRawData(directory, sub_directories, rate, outputDirectory):
        paths = FileReader.ReadFilePaths(directory, sub_directories)
        grapher = Graph()
        data = FileReader.ReadByFileRate(paths,1,(0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), rate)
        for activity in data:
            grapher.plotDirectory(data[activity], 500, paths[activity], "max500", outputDirectory)

if __name__ == '__main__':
    # Enviornemnt constants
     # Enviornemnt constants
    directory = 'C:\\Users\\Stephanos\\Documents\\Dev\\ML\\CSCI5957MachineLearningPhase1\\rawData\\'
    outputDirectory = 'C:\\Users\\Stephanos\\Documents\\Dev\ML\\CSCI5957MachineLearningPhase1\\test\\'
    sub_directories = ['Cycling', 'Driving', 'Running', 'Sitting', 'StairDown', 'StairUp', 'Standing', 'Walking']

    # -- Graph all the raw data --
    graphRawData(directory, sub_directories, 40, outputDirectory)

    paths = FileReader.ReadFilePaths(directory, sub_directories)
    grapher = Graph()
    #data = FileReader.ReadByFileRate(paths,1,(0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), 40)
    #grapher.plotDirectory(data, 500, paths, "max500", outputDirectory)

    
    #result = Feature.getChunkData(df)
    #print(result)
    # -- Synchronizing data  -- NEED to be split up by type first!
    #dataFrames = Data.shoeDataToDataFrame(data['Cycling'])
    #dataFrames = Synchronization.trim_start_end_times(dataFrames)
    #dataFrames = Synchronization.join_event_data(dataFrames)
    #example = Synchronization.chunkify_data_frame(dataFrames[0])
    #print(example)

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