import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
class SVM:
    
    #Generates a classification based on user input
    #dataFrame:pandas.DataFrame - combined dataframe with all activities on one frame
    #CValue:int - the value of C for the kernel to use in classification to control boundary tightness
    #kernelToUse:string - a designation for the kernel for the model to use... one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
    #testValuePercent:int/float - the percentage of the data that should be devoted to the test set 
    #isFixed:bool - whether or not to use a random value in the test
    @staticmethod
    def classify(dataFrame, CValue, kernelToUse, testValuePercent, isFixed):
        X = dataFrame.iloc[:, 2:dataFrame.shape[1]]
        Y = dataFrame['Activity']
        if(isFixed):
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, shuffle = True, random_state = 42, test_size = float(testValuePercent/100))
        else:
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, shuffle = True, test_size = float(testValuePercent/100))
        svm_model_linear = SVC(kernel = kernelToUse, C = CValue).fit(X_train, Y_train) 
        svm_predictions = svm_model_linear.predict(X_test)

        # model accuracy for X_test   
        accuracy = svm_model_linear.score(X_test, Y_test) 

        # creating a confusion matrix 
        cm = confusion_matrix(Y_test, svm_predictions) 

        SVM.printStats(CValue, kernelToUse, testValuePercent, isFixed, accuracy, cm)
        return accuracy

    @staticmethod
    def testNIterations(dataFrame, CValue, kernelToUse, testValuePercent, nIterations):
        accuracyResults = []
        for test in range(0, nIterations):
            accuracyResults.append(classify(dataFrame, CValue, kernelToUse, testValuePercent, False))
            
        return aaccuracyResults

    @staticmethod
    def findAverage(resultArray):
        numResults = len(resultArray)
        return (float)(_sum(accuracyResults, numResults)/numResults)

    @staticmethod
    def printStats(CValue, kernelToUse, testValuePercent, isFixed, accuracy, confusionMatrix):
        print("    The accuracy for SVM with settings: ")
        print("\n    Kernel: "+kernelToUse)
        print("\n    C - Value: "+str(CValue))
        print("\n    Fixed seed: "+str(isFixed))
        print("\n    Test Set Percentage: "+str(testValuePercent))
        print("\n    is: "+str(accuracy))
        print("\n\n The confusion matrix is as follows: ")
        print(confusionMatrix)