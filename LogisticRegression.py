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

    #Splits data set with given test_size ratio
    @staticmethod
    def splitTrainTestSet(dataFrame, test_size):
        y = dataFrame['Activity']
        X = dataFrame.iloc[:, 2:dataFrame.shape[1]]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

        return X_train, X_test, y_train, y_test

    #Generates a classification 
    @staticmethod
    def classify(dataFrame, test_size=.2):
        # Split dataset
        X_train, X_test, y_train, y_test = LogReg.splitTrainTestSet(dataFrame, test_size)

        # Create and train the logistic model
        logreg = LogisticRegression(C=.01)
        logreg.fit(X_train, y_train)
        
        # Make predictions
        y_pred=logreg.predict(X_test)

        # Print results
        LogReg.printStats(y_test, y_pred)

        # Return accuracy rating
        return metrics.accuracy_score(y_test, y_pred) * 100

    #Generates 3 classifications and plots for varying regularization rates
    @staticmethod
    def classifyWithRegularizationRates(dataFrame):
        # Split dataset
        le = LabelEncoder()
        le.fit(dataFrame['Activity'].astype(str))

        y = le.transform(dataFrame['Activity'].astype(str))
        X = dataFrame.iloc[:, 2:dataFrame.shape[1]]

        # classify small against large digits
        y = (y > 4).astype(np.int)

        l1_ratio = 0.5  # L1 weight in the Elastic-Net regularization

        fig, axes = plt.subplots(3, 3)

        # Set regularization parameter
        for i, (C, axes_row) in enumerate(zip((1, 0.1, 0.01), axes)):
            # turn down tolerance for short training time
            clf_l1_LR = LogisticRegression(C=C, penalty='l1', tol=0.01, solver='saga')
            clf_l2_LR = LogisticRegression(C=C, penalty='l2', tol=0.01, solver='saga')
            clf_en_LR = LogisticRegression(C=C, penalty='elasticnet', solver='saga',
                                           l1_ratio=l1_ratio, tol=0.01)
            clf_l1_LR.fit(X, y)
            clf_l2_LR.fit(X, y)
            clf_en_LR.fit(X, y)

            coef_l1_LR = clf_l1_LR.coef_.ravel()
            coef_l2_LR = clf_l2_LR.coef_.ravel()
            coef_en_LR = clf_en_LR.coef_.ravel()

            # coef_l1_LR contains zeros due to the
            # L1 sparsity inducing norm

            sparsity_l1_LR = np.mean(coef_l1_LR == 0) * 100
            sparsity_l2_LR = np.mean(coef_l2_LR == 0) * 100
            sparsity_en_LR = np.mean(coef_en_LR == 0) * 100

            print("C=%.2f" % C)
            print("{:<40} {:.2f}%".format("Sparsity with L1 penalty:", sparsity_l1_LR))
            print("{:<40} {:.2f}%".format("Sparsity with Elastic-Net penalty:",
                                          sparsity_en_LR))
            print("{:<40} {:.2f}%".format("Sparsity with L2 penalty:", sparsity_l2_LR))
            print("{:<40} {:.2f}".format("Score with L1 penalty:",
                                         clf_l1_LR.score(X, y)))
            print("{:<40} {:.2f}".format("Score with Elastic-Net penalty:",
                                         clf_en_LR.score(X, y)))
            print("{:<40} {:.2f}".format("Score with L2 penalty:",
                                         clf_l2_LR.score(X, y)))

        plt.show()

    @staticmethod
    def getLearningCurve(dataFrame):
        le = LabelEncoder()
        le.fit(dataFrame['Activity'].astype(str))

        y = le.transform(dataFrame['Activity'].astype(str))
        X = dataFrame.iloc[:, 2:dataFrame.shape[1]]
        train_sizes = [1, 100, 250, 500, 750, 1000, 1250]

        train_sizes, train_scores, validation_scores = learning_curve(
            estimator = LogisticRegression(C=1),
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


