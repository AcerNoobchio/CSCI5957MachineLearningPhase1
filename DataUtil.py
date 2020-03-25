import numpy as np
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

