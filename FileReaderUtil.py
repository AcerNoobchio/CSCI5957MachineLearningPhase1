import numpy as np
import glob as files
import os

class FileReader:
    """description of class"""

    # =========================  Reading  ============================================
#Reads in features by file - Currently not working - need to refactor to return data instead of graphing it
    @staticmethod
    def ReadByFile(filePaths, rowsToSkip, colsToUse):
        data = {'Cycling': [], 'Driving': [], 'Running': [], 'Sitting': [], 'StairDown': [], 'StairUp': [], 'Standing': []}
        for dir in filePaths:
            for file in filePaths[dir]:
                data[dir].append(FileReader.ReadFile(file, rowsToSkip, colsToUse, 1))
        finalData = np.asarray(data)
        return data

    @staticmethod
    def ReadByFileEvent(filePaths, rowsToSkip, colsToUse):
        data = {'Cycling': [], 'Driving': [], 'Running': [], 'Sitting': [], 'StairDown': [], 'StairUp': [], 'Standing': []}
        tempEvent = []

        for dir in filePaths:
            #Collects data per event, scanning file for name. Up to 4 events per activity
            for event in range(1,5):   
                for file in filePaths[dir]:
                    if str(event) in os.path.split(file)[1]:
                        tempEvent.append(FileReader.ReadFile(file, rowsToSkip, colsToUse, 1))
                if len(tempEvent) > 0:
                    data[dir].append(tempEvent)
                    tempEvent = []
        finalData = np.asarray(data)
        return data

    @staticmethod
    def ReadByFileRate(filePaths, rowsToSkip, colsToUse, rate):
        data = {'Cycling': [], 'Driving': [], 'Running': [], 'Sitting': [], 'StairDown': [], 'StairUp': [], 'Standing': []}
        for dir in filePaths:
            for file in filePaths[dir]:
                data[dir].append(FileReader.ReadFile(file, rowsToSkip, colsToUse, rate))
        finalData = np.asarray(data)
        return data

    #Reads in the lines of a file, only reading in the only nth file where n is the passed-in rate
    @staticmethod
    def ReadFile(filename, skipRows, colsToUse, rate):
        i = 0
        data = []
        with open(filename) as f:
            for line in f:
                if i % rate == 0:
                    data.append(line)
                i+=1
            finalData = np.genfromtxt(data, delimiter=",", skip_header=skipRows, usecols = colsToUse)
            return finalData

    #collects all of the filenames in a given directory
    @staticmethod
    def ReadFilePaths(directory, sub_directories):
        fileNames = {'Cycling': [], 'Driving': [], 'Running': [], 'Sitting': [], 'StairDown': [], 'StairUp': [], 'Standing': []}
        dataFiles = {'Cycling': [], 'Driving': [], 'Running': [], 'Sitting': [], 'StairDown': [], 'StairUp': [], 'Standing': []}
        for sub_dir in sub_directories:
            dataFiles[sub_dir].extend(files.glob(directory+sub_dir+'\\*.csv'))
        for dir in dataFiles:
            for file in dataFiles[dir]:
                if file.find("left") > 0 or file.find("right") > 0:
                    fileNames[dir].append(file)

        return fileNames


