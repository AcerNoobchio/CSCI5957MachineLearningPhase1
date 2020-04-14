import numpy as np
import pandas as pd
import statistics as stats
from sklearn import datasets
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

class LogReg(object):
    """Provides multiclass logistic regression model support for classification"""

    #Generates a classification 
    @staticmethod
    def classify(dataFrame):
        # Split dataset
        X_train, X_test, y_train, y_test = LogReg.splitTrainTestSet(dataFrame, .2)

        # Create and train the logistic model
        logreg = LogisticRegression()
        logreg.fit(X_train, y_train)
        
        # Make predictions
        y_pred=logreg.predict(X_test)

        # Print results
        LogReg.printStats(y_test, y_pred)

        # Return accuracy rating
        return metrics.accuracy_score(y_test, y_pred) * 100

    @staticmethod
    def splitTrainTestSet(dataFrame, test_size):
        y = dataFrame['Activity']
        X = dataFrame.drop(columns=['Activity'])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

        return X_train, X_test, y_train, y_test

    @staticmethod
    def testNIterations(dataFrame, nIterations):
        accuracyResults = []
        for test in range(0, nIterations):
            accuracyResults.append(LogReg.classify(dataFrame))
            
        return accuracyResults

    #Finds the average of a passed array
    @staticmethod
    def findAverage(resultArray):
        numResults = len(resultArray)
        return (float)(stats._sum(resultArray)[1]/numResults)

    @staticmethod
    def printStats(y_test, y_pred):
        print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
        print("Precision:",metrics.precision_score(y_test, y_pred, average='micro'))
        print("Recall:",metrics.recall_score(y_test, y_pred, average='micro'))


