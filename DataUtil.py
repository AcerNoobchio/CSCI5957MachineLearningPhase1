import numpy as np
import pandas as pd

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

    #Builds a labled pandas dataframe out of multidimesnional numpy array
    def shoeDataToDataFrame(data):
        labels = [ "Time (Milliseconds)","P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "acX", "acY", "acZ"]
        dataFrames = list()
        for d in data:
            d = pd.DataFrame(d, columns=labels)
            dataFrames.append(d)

        return dataFrames

    #Builds a labled pandas dataframe out of multidimesnional numpy array
    def phoneDataToDataFrame(data):
        labels = [ "Time (Milliseconds)","ax", "ay", "az", "gx", "gy", "gz"]
        dataFrames = list()
        for d in data:
            d = pd.DataFrame(d, columns=labels)
            dataFrames.append(d)

        return dataFrames

    #Takes a single column of a pandas dataframe and converts it into a size(n, 2) numpy array using col 0 as the X-Axis
    def dataFrameColToNumpy(pandasFrame, col):
        newFrame = pd.DataFrame(data = pandasFrame.iloc[:,0])                                  #Generate new frame with time values
        newFrame.insert(1, pandasFrame.columns[col], pandasFrame.iloc[:,col])                    #Create a frame with time vs whatever is in specified column
        numpyCol = newFrame.to_numpy()
        return numpyCol