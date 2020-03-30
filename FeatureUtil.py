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
        return (float)(stats._sum(data[:,1])[1]/data.shape[0])

    @staticmethod
    def findSum(data):
        return (float)(stats._sum(data[:,1])[1])

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
    def findPeakInfo(data):                           
        time = 0.0
        times = []
        min = 999999999.9
        max = 0.0
        average = 0.0
        newData = Rescale.rescaleX(data)                #When we subtract, we want a standard unit of time
        peaks = sig.find_peaks(newData[:,1])[0]              #grab the indices of the peaks, which are located at index 0
        if  len(peaks) > 0:
            for i in range (0, len(peaks)):                      #The indices are in the array contained in element 
                time = newData[peaks[i]][0] - time               #Add the differences in time for each peak
                times.append(time)
                peakHeight = newData[peaks[i]][1]
                if(peakHeight > max):                            #Designwise - should be in their own methods, but I really don't want to write this loop 4 times
                      max = peakHeight
                if(peakHeight < min):
                      min = peakHeight
                average += peakHeight
            timeToPeak = (float)(stats._sum(times)[1]/len(times)) 
            average /= len(peaks)
        else:
            min = 0
            max = 0.0
            timeToPeak = 0

        peakFeatures = {"ttp":timeToPeak, "min":min,  "max":max, "avg":average}

        return peakFeatures              #Return the average time (in normalized x-axis units)

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
            if not((x2 - x1) == 0):
                slopes.append(abs((y2 - y1)/(x2 - x1)))     #Abs so the value doesn't even out - might change later, if there is a zero in denom - invalid - don't add
            y1 = y2
            x1 = x2
        return (float)(stats._sum(slopes)[1]/len(slopes))

    #Iterates through the files in the data frame, sending activities to be broken up
    #dataFrame - Dictionary
    @staticmethod
    def exportDataSetFeatures(dataFrames, directory):
        featureFrame = []
        for key, value in dataFrames.items():
            directoryToSend = directory+key+"\\"
            featureFrame.append(FeatureUtil.exportActivityFeatures(value, directoryToSend))
        return featureFrame

    #Iterates through the Events in the data frame, sending the chunks to be broken up
    #events - List
    @staticmethod
    def exportActivityFeatures(events, directory):
        eventFrame = []
        for key, value in events.items():
            directoryToSend = directory+key+"\\"
            eventFrame.append(FeatureUtil.exportShoeFeatures(value, directoryToSend))
        return eventFrame

    #Iterates through the Events in the data frame, sending the chunks to be broken up
    #event - List
    @staticmethod
    def exportShoeFeatures(Event, directory):
        shoeFrame = []
        for shoe in Event:
            if "Shoe" in directory:
                if shoe == 0:
                    directoryToSend = directory+"Left"
                else:
                    directoryToSend = directory+"Right"
              
            else:
                if shoe == 0:
                    directoryToSend = directory+"Acc"
                else:
                    directoryToSend = directory+"Gyro"
            directoryToSend +="\\"
            shoeFrame.append(FeatureUtil.exportEventFeatures(shoe, directoryToSend))
        return shoeFrame

    #Iterates through the chunks in the data frame, sending each chunk to be analyzed
    #chunks - List
    @staticmethod
    def exportEventFeatures(chunks, directory):
        chunkFrame = []
        i = 0
        for chunk in chunks:
            filePath = directory+"Event"+str(i)+".csv"
            newEventFeatures = FeatureUtil.exportChunkFeatures(chunk)
            chunkFrame.append(newEventFeatures)
            newEventFeatures.to_csv(filePath)      #Export the csv
            i+=1

        return chunkFrame

    #Iterates through the chunks in the data frame, sending each chunk to be analyzed
    #chunk - List of Dataframes
    @staticmethod
    def exportChunkFeatures(chunk):
        dataFrames = pd.DataFrame()
        for frame in chunk:
            dataFrames = dataFrames.append(FeatureUtil.getChunkDataAsRow(frame), ignore_index = True)
        return dataFrames

    #Iterates through the columns in the data frame in the chunk itself, returns a single row with all of the values of all of the features
    #chunk - Dataframe
    @staticmethod
    def getChunkDataAsRow(chunk):
        labels = [  "Max:", 
                    "Min:" ,
                    "Median:", 
                    "Mode:", 
                    "Sum:", 
                    "Standard Deviation:",
                    "Kurtosis:",
                    "Area Under Curve:",
                    "Average Slope:",
                    "Skewness:",
                    "Time To Peak:",
                    "Min Peak:",
                    "Max Peak:",
                    "Avg Peak:"]        #putting this here because I want this class to remain static
        featureList = []
        newLabels = Rescale.generateColPairings(chunk, labels)
        chunkFrame = pd.DataFrame()
        for col in range(1, len(chunk.columns)):
            currentColumns = chunk.columns
            if (chunk.iloc[:,col].value_counts().any() > 0):
                featureList.extend(FeatureUtil.calculateFeatures(Rescale.dataFrameColToNumpy(chunk,col))) #Generate all of the available features
            else:
                featureList.extend([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])             #Please clean the empty columns so I don't have to do this :(
        featureSeries = pd.Series(featureList, index = newLabels)
        chunkFrame = chunkFrame.append(featureSeries, ignore_index=True)
        return chunkFrame

    #Iterates through the columns in the data frame in the chunk itself, returns a standard dataframe
    #chunk - Dataframe
    @staticmethod
    def getChunkDataAsFrame(chunk):
        labels = [  "Max:", 
                    "Min:" ,
                    "Median:", 
                    "Mode:", 
                    "Sum:", 
                    "Standard Deviation:",
                    "Kurtosis:",
                    "Area Under Curve:",
                    "Average Slope:",
                    "Skewness:",
                    "Time To Peak:",
                    "Min Peak:",
                    "Max Peak:",
                    "Avg Peak:"]        #putting this here because I want this class to remain static
        featureList = []
        chunkFrame = pd.DataFrame(index = newLabels)
        for col in range(1, len(chunk.columns)):
            currentColumns = chunk.columns
            if (chunk.iloc[:,col].value_counts().any() > 0):
                featureList.extend(FeatureUtil.calculateFeatures(Rescale.dataFrameColToNumpy(chunk,col))) #Generate all of the available features
                chunkFrame.insert(col-1, currentColumns.values[col], featureList)  #Format the returned list into a data frame of one column and add it to the chunk's frame
        return chunkFrame

    #Calculates the features for a given column of data - returns in an array
    #column - A size (n ,2) numpy array
    @staticmethod
    def calculateFeatures(column):
        peakStats = FeatureUtil.findPeakInfo(column)
        features = [FeatureUtil.findMax(column),
            FeatureUtil.findMin(column),
            FeatureUtil.findMedian(column),
            FeatureUtil.findMode(column),
            FeatureUtil.findSum(column),
            FeatureUtil.findStdDev(column),
            FeatureUtil.findKurtosis(column),
            FeatureUtil.findAreaUnderCurve(column),
            FeatureUtil.findAvgSlope(column),
            FeatureUtil.findSkewness(column),
            peakStats["ttp"],
            peakStats["min"],
            peakStats["max"],
            peakStats["avg"]]

        return features

    #Formats a passed list as a column - Keeping it for now, although I think it may just be better to use the pd.insert and pass that information - not as cohesive, but more convienient
    @staticmethod
    def formatArrayAsFrameCol(featureList, colLabel):
        colLabels = [colLabel]
        labels = [  "Max", 
                    "Min" ,
                    "Median", 
                    "Mode", 
                    "Sum", 
                    "Standard Deviation",
                    "Kurtosis",
                    "Skewness",
                    "Area Under Curve",
                    "Time To Peak",
                    "Average Slope"]        #putting this here because I want this class to remain static

        newList = np.asarray(featureList)   #convert to numpy array so it can be transposed easier, inefficient but we're not mining bitcoin here
        newList = np.transpose(newList)     #comes in as a row, need to make it a column
        featureFrame = pd.DataFrame(data=featureList, index = labels, columns = colLabels)
        return featureFrame

