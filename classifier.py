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
        activityDFs = {'Cycling': [], 'Driving': [], 'Running': [], 'Sitting': [], 'StairDown': [], 'StairUp': [], 'Standing': [], 'Walking': []}
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
    sub_directories = ['Cycling', 'Driving', 'Running', 'Sitting', 'StairDown', 'StairUp', 'Standing', 'Walking']
    paths = FileReader.ReadFilePaths(directory, sub_directories)

    # -- Graph all the raw data --
    #graphRawData(paths, 40, outputDirectory)

    # -- Synchronizing data  -- NEED to be split up by type first!
    #activityDFs = synchronizeDataFromPaths(paths)

    # -- Testing Feature methods --  
    #   print("Min: "+Feature.findMin(newData))
   
    df = pd.DataFrame(
	[[21.0, 72.0, 67.0],
	[23.0, 78.0, 69.0],
	[32.0, 74.0, 56.0],
	[52.0, 54.0, 76.0]],
	columns=['a', 'b', 'c'])
    result = Feature.getChunkData(df)
    print(result)
    # -- Plotting features --
    #col = 2
    #data = FileReader.ReadByFileRate(paths,1,(0, 2), 40)
    #data = Data.rescale2D(data)
    #grapher.plotFeature(data, 500, labels[col], "by Feature", outputDirectory, col) #Works finally - looks awful, will need to pass in selected files