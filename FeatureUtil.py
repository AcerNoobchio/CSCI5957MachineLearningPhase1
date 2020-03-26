import numpy as np
import pandas as pd
import statistics as stats
import scipy.integrate as deriv
import scipy.stats as sciStats
import scipy.signal as sig
from DataUtil import DataUtil as Rescale
class FeatureUtil:
    """Assuming individual 2-minute spans of 2d data being passed in (time x feature)"""

    @staticmethod
    def findMax(data):
        return max(data[:,1])

    @staticmethod
    def findMin(data):
        return min(data[:,1])

    @staticmethod
    def findMedian(data):
        return stats.median(data[:,1])

    @staticmethod
    def findMode(data):
        return (stats._sum(data[:,1])[1]/data.shape[0])

    @staticmethod
    def findSum(data):
        return stats._sum(data[:,1])[1]

    @staticmethod
    def findStdDev(data):
        return stats.stdev(data[:,1])

    @staticmethod
    def findKurtosis(data):                             #Kurtosis = fourth central moment divided by square of variance - autocorrects for bias - operates along one dimension
        return sciStats.kurtosis(data[:,1])

    @staticmethod
    def findSkewness(data):
        return sciStats.skew(data[:,1])

    @staticmethod
    def findAreaUnderCurve(data):
        newData = Rescale.rescaleX(data)                #Normalize x so it can be used to compare data sets
        return deriv.trapz(newData[:,1], newData[:,0])  #Take the area under the curve using the x axis as sample points (using 1d array assumes taking area between curve and axis)

    @staticmethod
    def findTimeToPeak(data):                           
        time = 0.0
        times = []
        newData = Rescale.rescaleX(data)                #When we subtract, we want a standard unit of time
        peaks = sig.find_peaks(newData[:,1])[0]                 #grab the indices of the peaks, which are located at index 0
        for i in range (0, len(peaks)):                      #The indices are in the array contained in element 
            time = newData[peaks[i]][0] - time               #Add the differences in time for each peak
            times.append(time)
        return (stats._sum(times)[1]/len(times))              #Return the average time (in normalized x-axis units)

    @staticmethod
    def findAvgSlope(data):
        slopes = []
        newData = Rescale.rescaleX(data)                #Normalize x so it can be used to compare data sets
        y1 = newData[0,1]
        x1 = newData[0,0]
        y2 = 0.0
        x2 = 0.0
        for i in range(1, newData.shape[0]):
            x2 = newData[i,0]
            y2 = newData[i,1]
            slopes.append(abs((y2 - y1)/(x2 - x1)))     #Abs so the value doesn't even out - might change later
            y1 = y2
            x1 = x2
        return (stats._sum(slopes)[1]/len(slopes))

    #Iterates through the files in the data frame, sending activities to be broken up
    @staticmethod
    def exportFeatureFrame(dataFrame):
        return dataFrame

    #Iterates through the activities in the data frame, sending the events to be broken up
    @staticmethod
    def exportActivityFeatures(activities):
        return activities

    #Iterates through the Events in the data frame, sending the chunks to be broken up
    @staticmethod
    def exportEventFeatures(events):
        return events

    #Iterates through the chunks in the data frame, sending the lines to be broken up
    @staticmethod
    def exportChunkFeatures(chunks):
        return chunks

    #Calculates the features for a given line of data and returns the corresponding feature values
    @staticmethod
    def exportLineFeatures(chunk):
        return Chunk
