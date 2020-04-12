import numpy as np
import pandas as pd
from sklearn import preprocessing

class DataUtil:
    """Holds utility methods that serve to scale data for specific shapes"""
    # ======================  Re-scaling  ==================================

    #rescales a single data point
    @staticmethod
    def rescaleElt(x, min, max):
        range = max - min
        return ((x-min)/range)

    #used for rescaling a single numpy array
    @staticmethod
    def rescaleX(dataIn):
        max = dataIn[dataIn.shape[0]-1][0]
        min = dataIn[0][0]
        for i in range(0, len(dataIn)):
           dataIn[i,0] = DataUtil.rescaleElt(dataIn[i][0], min, max)   
        return dataIn

    #Used for rescaling an array of numpy arrays
    @staticmethod
    def rescale2D(dataIn):
        for i in range(0, len(dataIn)):
            dataIn[i] = DataUtil.rescaleX(dataIn[i])
        return dataIn

    @staticmethod
    def trimOutliersBySTD(dataFrame):
        factor = 4
        for col in dataFrame:
            upper_lim = dataFrame[col].mean () + dataFrame[col].std () * factor
            lower_lim = dataFrame[col].mean () - dataFrame[col].std () * factor
            dataFrame = dataFrame[(dataFrame[col] < upper_lim) & (dataFrame[col] > lower_lim)]

        return dataFrame

    @staticmethod
    def trimOutliersBySTD(dataFrame, factor):
        for col in dataFrame.iloc[:,1:]:
            upper_lim = dataFrame[col].mean () + dataFrame[col].std () * factor
            lower_lim = dataFrame[col].mean () - dataFrame[col].std () * factor
            dataFrame = dataFrame[(dataFrame[col] < upper_lim) & (dataFrame[col] > lower_lim)]

        return dataFrame

    @staticmethod
    def trimRowsWithZeroValues(dataFrame):
        for col in dataFrame:
            dataFrame = dataFrame[(dataFrame[col] != 0.0)]

        return dataFrame

    @staticmethod
    def cleanRows(dataFrame, factor):
        dataFrame = DataUtil.trimRowsWithZeroValues(dataFrame)
        dataFrame = DataUtil.trimOutliersBySTD(dataFrame, factor)

        return dataFrame

    #Builds a labled pandas dataframe out of multidimesnional numpy array
    def shoeDataToDataFrame(data):
        labels = [ "Time (Milliseconds)","P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "acX", "acY", "acZ"]
        dataFrames = list()
        for d in data:
            d = pd.DataFrame(d, columns=labels)
            cleanD = DataUtil.cleanRows(d, 4)
            dataFrames.append(cleanD)

        return dataFrames

    #Builds a labled pandas dataframe out of multidimesnional numpy array
    def phoneDataToDataFrame(data):
        labels = [[ "Time (Milliseconds)","ax", "ay", "az"], [ "Time (Milliseconds)","gx", "gy", "gz"]]
        dataFrames = list()

        for i in range(0,2):
            data[i] = pd.DataFrame(data[i], columns=labels[i])
            data[i] = DataUtil.cleanRows(data[i], 4)
            dataFrames.append(data[i])

        return dataFrames

    #Takes a single column of a pandas dataframe and converts it into a size(n, 2) numpy array using col 0 as the X-Axis
    def dataFrameColToNumpy(pandasFrame, col):
        newFrame = pd.DataFrame(data = pandasFrame.iloc[:,0])                                  #Generate new frame with time values
        newFrame.insert(1, pandasFrame.columns[col], pandasFrame.iloc[:,col])                    #Create a frame with time vs whatever is in specified column
        numpyCol = newFrame.to_numpy()
        return numpyCol

    #Takes a chunk and a list and creates a pairing of each label to each column of the chunk
    #dataframe - pandas dataframe
    #Labels - list of strings
    @staticmethod
    def generateColPairings(dataframe, labels):
        newLabels = []
        for col in dataframe.columns.values:
            if not("Milliseconds" in col):
                for i in range(0, len(labels)):
                    newLabels.append(col +" "+ labels[i])
        return newLabels

    def minMaxScaleDataFrame(df):
        x = df.values #returns a numpy array
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        df = pd.DataFrame(x_scaled)
        return df


    @staticmethod
    def combineEventFeatures(activityFeatures):
            combinedFeatures = {'Cycling': [], 'Driving': [], 'Running': [], 'Sitting': [], 'StairDown': [], 'StairUp': [], 'Standing': []}
            leftFeatures = {'Cycling': [], 'Driving': [], 'Running': [], 'Sitting': [], 'StairDown': [], 'StairUp': [], 'Standing': []}
            rightFeatures = {'Cycling': [], 'Driving': [], 'Running': [], 'Sitting': [], 'StairDown': [], 'StairUp': [], 'Standing': []}

            #calculate dictionaries based on what is read in - this will keep us from having to change this method when we delete/include new data *crosses fingers*
            for activityLabel, activity in activityFeatures["Phone"].items():
                for event in range(0, len(activity['Gyro'])):
                    combinedFeatures[activityLabel].append(0)
                    leftFeatures[activityLabel].append(0)
                    rightFeatures[activityLabel].append(0)
            eventNum = -1


            #Merge the Phone data into the data frame and collect the left and right shoe data together to be merged in the next loop
            for dataTypeLabel, dataTypeList in activityFeatures.items():
                for activityLabel, activity in dataTypeList.items():
                    for dataSourceLabel, dataSource in activity.items():
                        eventNum = -1
                        for eventFile in dataSource:
                            eventNum += 1
                            if "Gyro" in dataSourceLabel or "Acc" in dataSourceLabel:
                                if isinstance(combinedFeatures[activityLabel][eventNum], int):
                                    combinedFeatures[activityLabel][eventNum] = eventFile
                                else:
                                    combinedFeatures[activityLabel][eventNum] = pd.concat([combinedFeatures[activityLabel][eventNum], eventFile], sort=False) #Right append
                            else:
                                if "Right" in dataSourceLabel:
                                    if isinstance(rightFeatures[activityLabel][eventNum], int):
                                        rightFeatures[activityLabel][eventNum] = eventFile
                                    else:
                                        rightFeatures[activityLabel][eventNum] = pd.merge(rightFeatures[activityLabel][eventNum], eventFile)   #Not anticipated to be used, here just in case 
                                else:
                                    if isinstance(leftFeatures[activityLabel][eventNum], int):
                                        leftFeatures[activityLabel][eventNum] = eventFile
                                    else:
                                        leftFeatures[activityLabel][eventNum] = pd.merge(leftFeatures[activityLabel][eventNum], eventFile) 
            
            #Merge left and right shoes together
            for activity, event in rightFeatures.items():
                eventNum = 0
                for rightFeature in event:
                    #Give left and right proper defining labels - need to move up the process so we can just add while reading in, but I'm tired and its 4am, so here it is
                    labelIndex = 0
                    for label in rightFeature.columns.values.tolist():
                        rightFeature.columns.values[labelIndex] = "right "+rightFeature.columns.values.tolist()[labelIndex]
                        labelIndex+=1
            
                    labelIndex = 0
                    for label in leftFeatures[activity][eventNum].columns.values.tolist():
                        leftFeatures[activity][eventNum].columns.values[labelIndex] = "left "+leftFeatures[activity][eventNum].columns.values.tolist()[labelIndex]
                        labelIndex+=1
            
                    rightFeature = pd.concat([leftFeatures[activity][eventNum], rightFeature], axis=1) #Combine the shoe data horizontally
                    rightFeatures[activity][eventNum] = rightFeature
                    eventNum += 1

            #Merge the shoe and phone data together  
            for activity, event in combinedFeatures.items():
                eventNum = 0
                for frameToCombine in event:
                    #frameToCombine = pd.merge(frameToCombine, rightFeatures[activity][eventNum], how ='outer') #Combine all of the data horizontally
                    frameToCombine = pd.concat([frameToCombine, rightFeatures[activity][eventNum]], axis=0, ignore_index = True) #Combine all of the data horizontally
                    combinedFeatures[activity][eventNum] = frameToCombine
                eventNum += 1
                                                   
            return combinedFeatures