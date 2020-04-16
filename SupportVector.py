import numpy as np
import pandas as pd
import statistics as stats
from sklearn import datasets
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
class SVM:
    
    #Generates a classification based on user input
    #dataFrame:pandas.DataFrame - combined dataframe with all activities on one frame
    #CValue:int - the value of C for the kernel to use in classification to control boundary tightness
    #kernelToUse:string - a designation for the kernel for the model to use... one of ‘linear’, ‘poly’, ‘rbf’ - Guassian, ‘sigmoid’, ‘precomputed’ - requires additional args
    #testValuePercent:int/float - the percentage of the data that should be devoted to the test set 
    #isFixed:bool - whether or not to use a random value in the test
    #printResults:bool - whether or not to print statistics for each run
    @staticmethod
    def classify(dataFrame, CValue, sValue, kernelToUse, testValuePercent, isFixed, printResults):
        
        #gammaVal = sValue/dataFrame.shape[1]
        X_train, X_test, Y_train, Y_test = SVM.splitTestData(dataFrame, testValuePercent, isFixed)   #Split the data

        svm_model_linear = SVC(kernel = kernelToUse, C = CValue).fit(X_train, Y_train)      #Generate model
        svm_predictions = svm_model_linear.predict(X_test)

        # model accuracy for X_test   
        accuracy = svm_model_linear.score(X_test, Y_test) 

        if(printResults):
            SVM.printStats(CValue, kernelToUse, testValuePercent, isFixed, accuracy, Y_test, svm_predictions)

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

    #Finds the accuracy values given a number of classification tests
    @staticmethod
    def testNIterations(dataFrame, CValue, sValue, kernelToUse, testValuePercent, nIterations):
        accuracyResults = []
        for test in range(0, nIterations):
            accuracyResults.append(SVM.classify(dataFrame, CValue, sValue, kernelToUse, testValuePercent, False, False))
            
        return accuracyResults

    #Collects the averages of C values in a range from 1 to the passed number of Cs to test
    @staticmethod
    def findCsUpToN(dataFrame, numCs, sToUse, kernelToUse, testValuePercent, iterationsPerTest):
        cAverages = []
        cAverages.append(-1);
        for cTest in range(1, numCs+1):
            accuracyResults = SVM.testNIterations(dataFrame, cTest, sToUse, kernelToUse, testValuePercent, iterationsPerTest)
            cAverages.append(SVM.findAverage(accuracyResults))
        return cAverages

    #Collects the averages of S values in a range from 1 to the passed number of Ss to test
    @staticmethod
    def findSsUpToN(dataFrame, cToUse, numSs, kernelToUse, testValuePercent, iterationsPerTest):
        cAverages = []
        cAverages.append(-1);
        for sTest in range(1, numSs+1):
            accuracyResults = SVM.testNIterations(dataFrame, cToUse, numSs, kernelToUse, testValuePercent, iterationsPerTest)
            cAverages.append(SVM.findAverage(accuracyResults))
        return cAverages

    #Finds the average of a passed array
    @staticmethod
    def findAverage(resultArray):
        numResults = len(resultArray)
        return (float)(stats._sum(resultArray)[1]/numResults)

    @staticmethod
    def printStats(CValue, kernelToUse, testValuePercent, isFixed, accuracy, Y_test, svm_predictions):
        print("    The results for for SVM with settings: ")
        print("\n    Kernel: "+kernelToUse)
        print("\n    C - Value: "+str(CValue))
        print("\n    Fixed seed: "+str(isFixed))
        print("\n    Test Set Percentage: "+str(testValuePercent))
        print("\n    are as follows: ")
        print("\n    Accuracy: "+str(accuracy))
        #print("\n    Precision: ",metrics.precision_score(Y_test, svm_predictions, average='micro'))
        #print("\n    Recall: ",metrics.recall_score(Y_test, svm_predictions, average='micro'))

        report_lr = metrics.precision_recall_fscore_support(Y_test, svm_predictions, average='micro')
        print ("\n     precision = %0.2f, recall = %0.2f, F1 = %0.2f, accuracy = %0.2f\n" % \
           (report_lr[0], report_lr[1], report_lr[2], metrics.accuracy_score(Y_test, svm_predictions)))

        print("\n    Accuracy: ",metrics.accuracy_score(Y_test, svm_predictions))
        print("\n    Precision: ",metrics.precision_score(Y_test, svm_predictions, average='weighted'))

        print("\n    Recall:",metrics.recall_score(Y_test, svm_predictions, average='weighted'))
        print("\n\n  Confusion Matrix: ")
        print(confusion_matrix(Y_test, svm_predictions))

    @staticmethod
    def getValidationCurve(dataFrame, cVal, kernelToUse, directory, filename):
        le = LabelEncoder()
        le.fit(dataFrame['Activity'].astype(str))

        y = le.transform(dataFrame['Activity'].astype(str))
        X = dataFrame.iloc[:, 2:dataFrame.shape[1]]

        param_range = np.logspace(-6, -1, 5)
        train_scores, test_scores = validation_curve(
            SVC(cVal, kernelToUse), X, y, param_name="gamma", param_range=param_range,
            scoring="accuracy", n_jobs=1)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        plt.title("Validation Curve with SVM")
        plt.xlabel(r"$\gamma$")
        plt.ylabel("Score")
        plt.ylim(0.0, 1.1)
        lw = 2
        plt.semilogx(param_range, train_scores_mean, label="Training score",
                     color="darkorange", lw=lw)
        plt.fill_between(param_range, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.2,
                         color="darkorange", lw=lw)
        plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
                     color="navy", lw=lw)
        plt.fill_between(param_range, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.2,
                         color="navy", lw=lw)
        plt.legend(loc="best")
        plt.savefig(directory+filename)

    @staticmethod
    def getLearningCurve(dataFrame, cVal, kernelToUse, directory, filename):
        le = LabelEncoder()
        le.fit(dataFrame['Activity'].astype(str))

        y = le.transform(dataFrame['Activity'].astype(str))
        X = dataFrame.iloc[:, 2:dataFrame.shape[1]]
        train_sizes = [1, 100, 250, 500, 750, 1000, 1250]

        train_sizes, train_scores, validation_scores = learning_curve(
            SVC(cVal, kernelToUse), X, y, train_sizes = train_sizes, cv=5,	
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
        plt.title('Learning curves for a SVM model', fontsize = 18, y = 1.03)
        plt.legend()
        plt.ylim(0,3)

        plt.savefig(directory+filename)