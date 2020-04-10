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
    colsPerDataType = {'Shoe': (0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), 'Phone': (1, 4, 6, 8)}
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
    colsPerDataType = {'Shoe': (0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), 'Phone': (1, 4, 6, 8)}

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

def getFeatureRankings(activityFeatures):
    rankedFeatures = {"Left": [], "Right": [], "Acc": [], "Gyro": []}
    leftShoeData = []
    rightShoeData = []
    accData = []
    gyroData = []


    firstTimeThrough = False

    for dataType in activityFeatures:
        firstTimeThrough = not firstTimeThrough
        for activity in dataType:
            for dataSource in activity:
                    if firstTimeThrough:
                        leftShoeData.append(dataSource[0])
                        rightShoeData.append(dataSource[1])
                    else:
                        accData.append(dataSource[0])
                        gyroData.append(dataSource[1])

    rankedFeatures["Left"] = Feature.rankFeatures(leftShoeData)
    rankedFeatures["Right"] = Feature.rankFeatures(rightShoeData)
    rankedFeatures["Acc"] = Feature.rankFeatures(accData)
    rankedFeatures["Gyro"] = Feature.rankFeatures(gyroData)

    return rankedFeatures

#This doesn't work, need to convert properly from list to dataframe
def plotRankedFeaturesByType(rankedFeatures, outputDirectory, numFeatures):
    for key in rankedFeatures:
        df = pd.DataFrame(rankedFeatures[key])
        #Graph.plotRankedFeatures(df.head(numFeatures), (key+"RankedFeaturesGraph"), outputDirectory)

if __name__ == '__main__':
    # Set up enviornemnt constants and read in file paths
    print("Setting up enviornment and collecting paths to raw data files\n")
    #directory = 'C:\\Users\\Stephanos\\Documents\\Dev\\ML\\CSCI5957MachineLearningPhase1\\rawData\\'
    #outputDirectory = 'C:\\Users\\Stephanos\\Documents\\Dev\ML\\CSCI5957MachineLearningPhase1\\test\\'
    #featureDirectory = 'C:\\Users\\Stephanos\\Documents\\Dev\ML\\CSCI5957MachineLearningPhase1\\featureData\\'
    directory = 'C:\\Users\\jacob\\source\\repos\\MachineLearningPhase1\\MachineLearningPhase1\\rawData\\'
    outputDirectory = 'C:\\Users\\jacob\\source\\repos\\MachineLearningPhase1\\MachineLearningPhase1\\test\\'
    featureDirectory = 'C:\\Users\\jacob\\source\\repos\\MachineLearningPhase1\\MachineLearningPhase1\\featureData\\'
    sub_directories = ['Cycling', 'Driving', 'Running', 'Sitting', 'StairDown', 'StairUp', 'Standing']
    parent_directories = ['Phone', 'Shoe']
    #paths = FileReader.ReadFilePaths(directory, sub_directories)
    
    # -- Graph all the raw data --
    #print("Graphing all the raw data\n")
    #graphRawData(paths, 40, outputDirectory)
    #print("Finished graphing raw data\n")

    # -- Synchronizing data
    #print("Synchronizing and cleaning raw data... This could take a sec\n")
    #allDataDFs = synchronizeDataFromPaths(paths)
    #print("Finished synchronizing/cleaning raw data\n")
    
    # -- Generate features for each chunk of data, saving in .csv files --  
    #print("Extraplating and saving features for cleaned data... This will take a sec\n")
    #features = Feature.exportDataSetFeatures(allDataDFs, featureDirectory)
    #print("Finished saving feature files\n")

    # -- Plotting features (currently non-functional) --
    #col = 2
    #data = FileReader.ReadByFileRate(paths,1,(0, 2), 40)
    #data = Data.rescale2D(data)
    #grapher.plotFeature(data, 500, labels[col], "by Feature", outputDirectory, col) #Works finally - looks awful, will need to pass in selected files

    # -- Reading Features -- 
    print("Loading feature Data....\n")
    paths = FileReader.ReadFeaturePaths(featureDirectory, parent_directories, sub_directories)
    features = FileReader.ReadByFileFeatures(paths, 0)
    print("Feature Data Sucessfully Loaded\n")
    # -- Ranking features --
    #print("Ranking features by data type\n")
    #rankedFeatures = getFeatureRankings(features)
    #plotRankedFeaturesByType(rankedFeatures, featureDirectory, 10)
    #print("Finished ranking features\n")