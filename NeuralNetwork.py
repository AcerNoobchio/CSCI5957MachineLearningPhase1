import numpy as np
import pandas as pd
import statistics as stats
from sklearn import datasets
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
class NeuralNetwork(object):
    """description of class"""

        #Generates a classification based on user input
    #dataFrame:pandas.DataFrame - combined dataframe with all activities on one frame
    #CValue:int - the value of C for the kernel to use in classification to control boundary tightness
    #kernelToUse:string - a designation for the kernel for the model to use... one of ‘linear’, ‘poly’, ‘rbf’ - Guassian, ‘sigmoid’, ‘precomputed’ - requires additional args
    #testValuePercent:int/float - the percentage of the data that should be devoted to the test set 
    #isFixed:bool - whether or not to use a random value in the test
    #printResults:bool - whether or not to print statistics for each run
    @staticmethod
    def classify(dataFrame, alphaToUse, hiddenLayerSize, solverToUse, testValuePercent, isFixed, printResults):
        
        X_train, X_test, Y_train, Y_test = NeuralNetwork.splitTestData(dataFrame, testValuePercent, isFixed)      #Split the data

        mlp = MLPClassifier(activation='logistic', hidden_layer_sizes = hiddenLayerSize, solver=solverToUse, alpha=alphaToUse) #Generate the Learning infrastructure

        mlp_model = mlp.fit(X_train, Y_train)                                                           #generate model from training data
        mlp_predictions = mlp_model.predict(X_test)                                                     #Make predictions

        accuracy = mlp_model.score(X_test, Y_test)                                                #Model Accuracy

        if(printResults):
            NeuralNetwork.printStats(alphaToUse, solverToUse, hiddenLayerSize, testValuePercent, isFixed, accuracy, Y_test, mlp_predictions)

        return accuracy*100

    @staticmethod
    def splitTestData(dataFrame, testValuePercent, isFixed):
        X = dataFrame.iloc[:, 2:dataFrame.shape[1]]
        Y = dataFrame['Activity']
        if(isFixed):    #Use the same seed when generating test and training sets
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, shuffle = True, random_state = 42, test_size = float(testValuePercent/100))
        else:           #Use a completely random set of test and training data
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, shuffle = True, test_size = float(testValuePercent/100))

        return X_train, X_test, Y_train, Y_test

    @staticmethod
    def printStats(Alpha, solverToUse, hiddenLayerSize, testValuePercent, isFixed, accuracy, Y_test, mlp_predictions):
        print("    The results for for SVM with settings: ")
        print("\n    Solver: "+solverToUse)
        print("\n    Alpha: "+str(Alpha))
        print("\n    Hidden Layer Dimensions: "+str(hiddenLayerSize))
        print("\n    Fixed seed: "+str(isFixed))
        print("\n    Test Set Percentage: "+str(testValuePercent))
        print("\n    are as follows: ")
        print("\n    Accuracy: "+str(accuracy))
        #print("\n    Precision: ",metrics.precision_score(Y_test, svm_predictions, average='micro'))
        #print("\n    Recall: ",metrics.recall_score(Y_test, svm_predictions, average='micro'))

        report_lr = metrics.precision_recall_fscore_support(Y_test, mlp_predictions, average='micro')
        print ("\n     precision = %0.2f, recall = %0.2f, F1 = %0.2f, accuracy = %0.2f\n" % \
           (report_lr[0], report_lr[1], report_lr[2], metrics.accuracy_score(Y_test, mlp_predictions)))

        print("\n    Accuracy: ",metrics.accuracy_score(Y_test, mlp_predictions))
        print("\n    Precision: ",metrics.precision_score(Y_test, mlp_predictions, average='weighted'))

        print("\n    Recall:",metrics.recall_score(Y_test, mlp_predictions, average='weighted'))
        print("\n\n  Confusion Matrix: ")
        print(confusion_matrix(Y_test, mlp_predictions))