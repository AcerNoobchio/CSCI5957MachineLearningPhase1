import numpy as np
import statistics as stats
import scipy.integrate as deriv
import scipy.stats as sciStats
import scipy.signal as sig
from RescalingUtil import RescalingUtil as Rescale
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
        return stats.mode(data[:,1])

    @staticmethod
    def findSum(data):
        return stats.sum(data[:,1])

    @staticmethod
    def findStdDev(data):
        return stats.stdev(data[:,1])

    @staticmethod
    def findKurtosis(data):                             #Kurtosis = fourth central moment divided by square of variance - autocorrects for bias - operates along one dimension
        return sciStats.kurtosis(data[:,1])

    @staticmethod
    def findSkewness(data):
        return sciStats.findSkewness(data[:,1])

    @staticmethod
    def findAreaUnderCurve(data):
        newData = Rescale.rescaleX(data)                #Normalize x so it can be used to compare data sets
        return deriv.trapz(newData[:,0], newData[:,1])  #Take the area under the curve using the x axis as sample points (using 1d array assumes taking area between curve and axis)

    @staticmethod
    def findTimeToPeak(data):                           
        time = 0.0
        times = []
        newData = Rescale.rescaleX(data)                #When we subtract, we want a standard unit of time
        peaks = sig.find_peaks(newData[:,1])            #grab the indices of the peaks
        for peak in preaks:
            time = time - newData[peak,0]               #Add the differences in time for each peak
            times.append(time)
        return (stats.sum(time)/len(time))              #Return the average time (in normalized x-axis units)

    @staticmethod
    def findAvgSlope(data):
        y1 = 0.0
        x1 = 0.0
        y2 = 0.0
        x2 = 0.0
        slopes = []
        newData = Rescale.rescaleX(data)                #Normalize x so it can be used to compare data sets
        for i in range(1, newData.shape[0]):
            x2 = newData[i,0]
            y2 = newData[i,1]
            slopes.append(abs((y2 - y1)/(x2 - x1)))     #Abs so the value doesn't even out - might change later
            y1 = y2
            x1 = x2
        return (stats.sum(slopes)/len(slopes))
