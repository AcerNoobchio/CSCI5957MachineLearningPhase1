import pandas as pd
from ClassifierUtil import ClassifierUtil
from FeatureUtil import FeatureUtil as Feature
from FileReaderUtil import FileReader
from SupportVector import SVM
from LogisticRegression import LogReg
from GraphUtil import GraphUtil as Graph
import os

if __name__ == '__main__':
    # -- Create Instance of helper class --
    classifierUtil = ClassifierUtil()

    # -- Set up enviornemnt constants and read in file paths --
    print("Setting up enviornment and collecting paths to raw data files\n")
    #directory = 'C:\\Users\\Stephanos\\Documents\\Dev\\ML\\CSCI5957MachineLearningPhase1\\rawData\\'
    #outputDirectory = 'C:\\Users\\Stephanos\\Documents\\Dev\ML\\CSCI5957MachineLearningPhase1\\test\\'
    #featureDirectory = 'C:\\Users\\Stephanos\\Documents\\Dev\ML\\CSCI5957MachineLearningPhase1\\featureData\\'
    #combinedFeatureDirectory = 'C:\\Users\\Stephanos\\Documents\\Dev\ML\\CSCI5957MachineLearningPhase1\\combinedFeatureData\\'
    
    directory = 'C:\\Users\\jacob\\source\\repos\\MachineLearningPhase1\\MachineLearningPhase1\\rawDataOriginal\\'
    outputDirectory = 'C:\\Users\\jacob\\source\\repos\\MachineLearningPhase1\\MachineLearningPhase1\\test\\'
    featureDirectory = 'C:\\Users\\jacob\\source\\repos\\MachineLearningPhase1\\MachineLearningPhase1\\featureData\\'
    combinedFeatureDirectory = 'C:\\Users\\jacob\\source\\repos\\MachineLearningPhase1\\MachineLearningPhase1\\combinedFeatureData\\'
    
    paths = classifierUtil.getRawDataFilePaths(directory)
    grapher = Graph()
    # -- Graph all the raw data --
    #print("Graphing all the raw data\n")
    #classifierUtil.graphRawData(paths, 40, outputDirectory)
    #print("Finished graphing raw data\n")

    # -- Synchronizing data --
    #print("Synchronizing and cleaning raw data... This could take a sec\n")
    #allDataDFs = classifierUtil.synchronizeDataFromPaths(paths)
    #print("Finished synchronizing/cleaning raw data\n")
    
    # -- Generate features for each chunk of data, saving in .csv files --  
    #print("Extraplating and saving features for cleaned data... This will take a sec\n")
    #features = Feature.exportDataSetFeatures(allDataDFs, featureDirectory)
    #print("Finished saving feature files\n")

    # -- Plotting features (currently non-functional) --
    #print("Plotting feature data...\n")
    #classifierUtil.plotFeatureData(paths, outputDirectory)
    #print("Finsihed plotting feature data.\n")

    # -- Ranking features --
    #print("Ranking features by data type\n")
    #rankedFeatures = classifierUtil.getFeatureRankings(features)
    #print("Finished ranking features\n")

    # -- Reading Features -- 
    #print("Loading feature Data....\n")
    #features = classifierUtil.readAllFeaturesFromPaths(featureDirectory)
    #print("Feature Data Sucessfully Loaded\n")

    # -- Combining Feature Data --
    #print("Combining and saving final feature data...\n")
    #allCombined = classifierUtil.combineAndSaveAllFeatures(features, combinedFeatureDirectory)
    #print("All features combined and saved as AllFiles.csv\n")

    # -- Load combined feature data to train models --
    print("Loading combined feature data... \n")
    allCombined = pd.read_csv(combinedFeatureDirectory+"AllFiles.csv")
    print("Combine Feature data loaded\n")

    # -- Train and classify with SVM --
    numCs = 50
    numSs = 50
    kernelToUse = 'rbf' #gaussian
    testValuePercent = 20
    iterationsPerTest = 20
    chosenC = 9
    chosenS = 1
    graphName = "C"+str(numCs)+"Kernel"+kernelToUse+"TestPct"+str(testValuePercent)+"Itrs"+str(iterationsPerTest)
    lcGraphName = "LearningCurveKernel"+kernelToUse+"C"+str(chosenC)+"TestPct"+str(testValuePercent)

    #This takes awhile - feel free to comment it out if I forget when I push

    #---- Testing C-Value ------
    #SVM.classify(allCombined, chosenC, chosenS, kernelToUse, testValuePercent, True, True)
    average = SVM.testNIterations(allCombined, chosenC, chosenS, kernelToUse, testValuePercent, 1000)
    print("Average: ", SVM.findAverage(average))
    #cRanks = SVM.findCsUpToN(allCombined, numCs, chosenS,kernelToUse, testValuePercent, iterationsPerTest)
    #bestAccuracy = max(cRanks[1:])
    #worstAccuracy = min(cRanks[1:])
    #print("C Value with best Accuracy: ",cRanks.index(bestAccuracy), " Accuracy: " + str(bestAccuracy))
    #print("C Value with worst Accuracy: ",cRanks.index(worstAccuracy), " Accuracy: " + str(worstAccuracy))
    #grapher.plotArray(cRanks, 100, 1, "C-Value","Accuracy", "sRanking", graphName, outputDirectory)

    #---- Testing sigma2rd Value ------
    #sRanks = SVM.findSsUpToN(allCombined, chosenC, numSs,kernelToUse, testValuePercent, iterationsPerTest)
    #bestAccuracy = max(sRanks[1:])
    #worstAccuracy = min(sRanks[1:])
    #print("S Value with best Accuracy: ",sRanks.index(bestAccuracy), " Accuracy: " + str(bestAccuracy))
    #print("S Value with worst Accuracy: ",sRanks.index(worstAccuracy), " Accuracy: " + str(worstAccuracy))

    #grapher.plotArray(sRanks, 100, 1, "S-Value","Accuracy", "sRanking", graphName, outputDirectory)
    #SVM.getLearningCurve(allCombined, chosenC, kernelToUse, outputDirectory, lcGraphName)
    #SVM.getValidationCurve(allCombined, chosenC, kernelToUse, outputDirectory, lcGraphName)
    # -- Train and classify with SVM --
    #LogReg.getLearningCurve(allCombined)