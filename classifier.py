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

if __name__ == '__main__':
    # Enviornemnt constants
     # Enviornemnt constants
    directory = 'C:\\Users\\Stephanos\\Documents\\Dev\ML\\CSCI5957MachineLearningPhase1\\rawData\\'
    outputDirectory = 'C:\\Users\\Stephanos\\Documents\\Dev\ML\\CSCI5957MachineLearningPhase1\\test\\'

    # -- Read paths of all data files, in all sub_directories --
    #sub_directories = ['Cycling', 'Driving', 'Running', 'Sitting', 'StairDown', 'StairUp', 'Standing', 'Walking']
    #paths = FileReader.ReadFilePaths(directory, sub_directories)
    #grapher = Graph()
    #data = FileReader.ReadByFile(paths,1,(0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12))
    #grapher.plotDirectory(data, 500, paths, "max500", outputDirectory)

    df = pd.DataFrame(
	[[21, 72, 67],
	[23, 78, 69],
	[32, 74, 56],
	[52, 54, 76]],
	columns=['a', 'b', 'c'])
    result = Feature.getChunkData(df)
    print(result)
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