import numpy as np
import pandas as pd
import statistics as stats
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import LabelEncoder
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
    def getLearningCurve(dataFrame):
        le = LabelEncoder()
        le.fit(dataFrame['Activity'].astype(str))

        y = le.transform(dataFrame['Activity'].astype(str))
        X = dataFrame.iloc[:, 2:dataFrame.shape[1]]
        train_sizes = [1, 100, 250, 500, 750, 1000, 1250]

        train_sizes, train_scores, validation_scores = learning_curve(
            estimator = LogisticRegression(),
            X = X,
            y = y, train_sizes = train_sizes, cv = 5,
            scoring = 'neg_mean_squared_error',
            shuffle=True,)

        train_scores_mean = -train_scores.mean(axis = 1)
        validation_scores_mean = -validation_scores.mean(axis = 1)

        print('Mean training scores\n\n', pd.Series(train_scores_mean, index = train_sizes))
        print('\n', '-' * 20) # separator
        print('\nMean validation scores\n\n',pd.Series(validation_scores_mean, index = train_sizes))
        
        plt.style.use('seaborn')
        plt.plot(train_sizes, train_scores_mean, label = 'Training error')
        plt.plot(train_sizes, validation_scores_mean, label = 'Validation error')
        plt.ylabel('MSE', fontsize = 14)
        plt.xlabel('Training set size', fontsize = 14)
        plt.title('Learning curves for a logistic regression model', fontsize = 18, y = 1.03)
        plt.legend()
        plt.ylim(0,3)

        plt.show()
    @staticmethod
    def splitTrainTestSet(dataFrame, test_size):
        y = dataFrame['Activity']
        X = dataFrame.iloc[:, 2:dataFrame.shape[1]]
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

        report_lr = metrics.precision_recall_fscore_support(y_test, y_pred, average='micro')
        print ("\nprecision = %0.2f, recall = %0.2f, F1 = %0.2f, accuracy = %0.2f\n" % \
           (report_lr[0], report_lr[1], report_lr[2], metrics.accuracy_score(y_test, y_pred)))

        print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
        print("Precision:",metrics.precision_score(y_test, y_pred, average='weighted'))
        print("Recall:",metrics.recall_score(y_test, y_pred, average='weighted'))


