import numpy as np
import pandas as pd
import glob as files
import os

class FileReader:
    """description of class"""

    # =========================  Reading  ============================================
#Reads in features by file - Currently not working - need to refactor to return data instead of graphing it
    @staticmethod
    def ReadByFile(filePaths, rowsToSkip, colsToUse):
        data = {'Cycling': [], 'Driving': [], 'Running': [], 'Sitting': [], 'StairDown': [], 'StairUp': [], 'Standing': [], 'Walking': []}
        for dir in filePaths:
            for file in filePaths[dir]:
                data[dir].append(FileReader.ReadFile(file, rowsToSkip, colsToUse, 1))
        finalData = np.asarray(data)
        return data

    @staticmethod
    def ReadByFileEvent(filePaths, rowsToSkip, colsToUse):
        data = {'Cycling': [], 'Driving': [], 'Running': [], 'Sitting': [], 'StairDown': [], 'StairUp': [], 'Standing': [], 'Walking': []}
        tempEvent = []

        accCols = (1, 4, 6, 8)
        gyroCols = (1, 10, 12, 14)
        #gyroCols = (1, 4, 6, 8)

        for dir in filePaths:
            #Collects data per event, scanning file for name. Up to 4 events per activity
            for event in range(1,5):   
                for file in filePaths[dir]:
                    if str(event) in os.path.split(file)[1]:
                        if "Acc" in file:
                            tempEvent.append(FileReader.ReadFile(file, rowsToSkip, accCols, 1))
                        elif "Gyro" in file:
                            tempEvent.append(FileReader.ReadFile(file, rowsToSkip, gyroCols, 1))
                        else:
                            tempEvent.append(FileReader.ReadFile(file, rowsToSkip, colsToUse, 1))
                if len(tempEvent) > 0:
                    data[dir].append(tempEvent)
                    tempEvent = []
        finalData = np.asarray(data)
        return data

    def ReadByFileFeatures(filePaths, rowsToSkip):
        data = {'Cycling': {}, 'Driving': {}, 'Running': {}, 'Sitting': {}, 'StairDown': {}, 'StairUp': {}, 'Standing': {}, 'Walking': {}}
        dataTypes = {'Phone' : {}, 'Shoe' : {}}
        phoneTypes = {'Acc': [], 'Gyro': []}
        shoeTypes = {'Left': [], 'Right': []}

        for dir in filePaths:
            #Collects data per event, scanning file for name. Up to 4 events per activity
            for event in range(0,5):   
                for folder, fileList in filePaths[dir].items():
                    for file in fileList:
                            if 'Gyro' in file:
                                phoneTypes['Gyro'].append(FileReader.ReadFileNative(file, rowsToSkip))
                            elif 'Acc' in file:
                                phoneTypes['Acc'].append(FileReader.ReadFileNative(file, rowsToSkip))
                            elif 'Left' in file:
                                shoeTypes['Left'].append(FileReader.ReadFileNative(file, rowsToSkip))
                            elif 'Right' in file:
                                shoeTypes['Right'].append(FileReader.ReadFileNative(file, rowsToSkip))

                    if len(phoneTypes['Gyro']) > 0 or len(phoneTypes['Acc']) > 0:
                        data[folder].update(phoneTypes)
                        phoneTypes = {'Acc': [], 'Gyro': []}
                    
                    if len(shoeTypes['Left']) > 0 or len(shoeTypes['Right']) > 0:
                        data[folder].update(shoeTypes)
                        shoeTypes = {'Left': [], 'Right': []}

                if len(data[folder]) > 0:
                    dataTypes[dir].update(data)
                    data = {'Cycling': {}, 'Driving': {}, 'Running': {}, 'Sitting': {}, 'StairDown': {}, 'StairUp': {}, 'Standing': {}, 'Walking': {}}
        return dataTypes

    @staticmethod
    def ReadByFileRate(filePaths, rowsToSkip, colsToUse, rate):
        data = {'Cycling': [], 'Driving': [], 'Running': [], 'Sitting': [], 'StairDown': [], 'StairUp': [], 'Standing': [], 'Walking': []}
        for dir in filePaths:
            for file in filePaths[dir]:
                data[dir].append(FileReader.ReadFile(file, rowsToSkip, colsToUse, rate))
        finalData = np.asarray(data)
        return data

    @staticmethod
    def ReadFileNative(filename, skipRows):
        labels = []
        data = []
        with open(filename) as f:
            i = 0
            for line in f:
                if(i == 0):
                    labels.append(line)
                else:
                    data.append(line)
                i+=1
            finalData = pd.DataFrame(data, columns = labels)
            return finalData


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

    def ReadFeaturePaths(directory, parent_directories, sub_directories):
        fileNames = {'Shoe': {'Cycling': [], 'Driving': [], 'Running': [], 'Sitting': [], 'StairDown': [], 'StairUp': [], 'Standing': [], 'Walking': []}, 
                     'Phone': {'Cycling': [], 'Driving': [], 'Running': [], 'Sitting': [], 'StairDown': [], 'StairUp': [], 'Standing': [], 'Walking': []}}
        dataFiles = {'Cycling': [], 'Driving': [], 'Running': [], 'Sitting': [], 'StairDown': [], 'StairUp': [], 'Standing': [], 'Walking': []}
        shoeDirectories = ["Left", "Right"]
        phoneDirectories = ["Acc", "Gyro"]

        for p_directory in parent_directories:
            for sub_dir in sub_directories:
                if "Phone" in p_directory:
                    for phone_dir in phoneDirectories:
                        dataFiles[sub_dir].extend(files.glob(directory+p_directory+"\\"+sub_dir+"\\"+phone_dir+'\\*.csv'))
                else:
                    for shoe_dir in shoeDirectories:
                        dataFiles[sub_dir].extend(files.glob(directory+p_directory+"\\"+sub_dir+"\\"+shoe_dir+'\\*.csv'))
            for dir in dataFiles:
                for file in dataFiles[dir]:
                    if os.path.isfile(file):
                        if file.find("Left") > 0 or file.find("Right") > 0:
                            fileNames['Shoe'][dir].append(file)
                        elif file.find("Acc") > 0 or file.find("Gyro") > 0:
                            fileNames['Phone'][dir].append(file)
            dataFiles = {'Cycling': [], 'Driving': [], 'Running': [], 'Sitting': [], 'StairDown': [], 'StairUp': [], 'Standing': [], 'Walking': []}

        return fileNames


    #collects all of the filenames in a given directory
    @staticmethod
    def ReadFilePaths(directory, sub_directories):
        fileNames = {'Shoe': {'Cycling': [], 'Driving': [], 'Running': [], 'Sitting': [], 'StairDown': [], 'StairUp': [], 'Standing': [], 'Walking': []}, 
                     'Phone': {'Cycling': [], 'Driving': [], 'Running': [], 'Sitting': [], 'StairDown': [], 'StairUp': [], 'Standing': [], 'Walking': []}}
        dataFiles = {'Cycling': [], 'Driving': [], 'Running': [], 'Sitting': [], 'StairDown': [], 'StairUp': [], 'Standing': [], 'Walking': []}
        for sub_dir in sub_directories:
            dataFiles[sub_dir].extend(files.glob(directory+sub_dir+'\\*.csv'))
        for dir in dataFiles:
            for file in dataFiles[dir]:
                if file.find("left") > 0 or file.find("right") > 0:
                    fileNames['Shoe'][dir].append(file)
                elif file.find("Acc") > 0 or file.find("Gyro") > 0:
                    fileNames['Phone'][dir].append(file)

        return fileNames


