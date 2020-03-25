import pandas as pd
from pandas import read_table 
import numpy as np
import datetime as dt
from StringUtil import StringUtil as String
from RescalingUtil import RescalingUtil as Rescale
from FileReaderUtil import FileReader
from GraphUtil import GraphUtil as Graph
try:
    # [OPTIONAL] Seaborn makes plots nicer
    import seaborn
except ImportError:
    pass

if __name__ == '__main__':
    labels = [ "Time (Milliseconds)", "Time" ,"P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9"]
    directory = 'C:\\Users\\Stephanos\\Documents\\Dev\ML\\CSCI5957MachineLearningPhase1\\rawData\\'
    outputDirectory = 'C:\\Users\\Stephanos\\Documents\\Dev\ML\\CSCI5957MachineLearningPhase1\\test\\'
    #Read paths of all data files, in all sub_directories 
    sub_directories = ['Cycling\\', 'Driving\\', 'Running\\', 'Sitting\\', 'StairDown\\', 'StairUp\\', 'Standing\\', 'Walking\\']
    paths = FileReader.ReadFilePaths(directory, sub_directories)
    grapher = Graph()
    #data = FileReader.ReadByFileRate(paths,1,(0, 2, 3, 4, 5, 6, 7, 8, 9, 10), 20)
    #Graph.plotDirectory(data, 500, paths, "max500", outputDirectory)
    col = 2
    data = FileReader.ReadByFileRate(paths,1,(0, 2), 40)
    data = Rescale.rescale2D(data)
    grapher.plotFeature(data, 500, labels[col], "by Feature", outputDirectory, col) #Works finally - looks awful, will need to pass in selected files