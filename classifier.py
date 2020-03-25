import pandas as pd
from pandas import read_table 
import numpy as np
import datetime as dt
from DataUtil import DataUtil as Data
from FileReaderUtil import FileReader
from GraphUtil import GraphUtil as Graph
from SynchronizationUtil import SynchronizationUtil as Synchronization
try:
    # [OPTIONAL] Seaborn makes plots nicer
    import seaborn
except ImportError:
    pass

if __name__ == '__main__':
    directory = 'C:\\Users\\Stephanos\\Documents\\Dev\ML\\CSCI5957MachineLearningPhase1\\rawData\\'
    outputDirectory = 'C:\\Users\\Stephanos\\Documents\\Dev\ML\\CSCI5957MachineLearningPhase1\\test\\'
    #Read paths of all data files, in all sub_directories 
    sub_directories = ['Cycling', 'Driving', 'Running', 'Sitting', 'StairDown', 'StairUp', 'Standing', 'Walking']
    paths = FileReader.ReadFilePaths(directory, sub_directories)
    grapher = Graph()
    data = FileReader.ReadByFileRate(paths,1,(0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), 20)
    #grapher.plotDirectory(data, 500, paths, "max500", outputDirectory)
    
    # Synchronizing data  -- NEED to be split up by type first!
    dataFrames = Data.shoeDataToDataFrame(data['Cycling'])
    print(dataFrames[0].head())
    #dataFrames = Synchronization.trim_start_end_times(dataFrames)
    #Synchronization.join_event_data(dataFrames)


    #Plotting features
    col = 2
    data = FileReader.ReadByFileRate(paths,1,(0, 2), 40)
    #data = Data.rescale2D(data)
    #grapher.plotFeature(data, 500, labels[col], "by Feature", outputDirectory, col) #Works finally - looks awful, will need to pass in selected files