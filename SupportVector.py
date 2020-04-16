import numpy as np
import pandas as pd
import statistics as stats
from sklearn import datasets
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

        #SVM.printStats(CValue, kernelToUse, testValuePercent, isFixed, accuracy, cm)
        return accuracy*100

    #Finds the accuracy values given a number of classification tests
    @staticmethod
    def testNIterations(dataFrame, CValue, kernelToUse, testValuePercent, nIterations):
        accuracyResults = []
        for test in range(0, nIterations):
            accuracyResults.append(SVM.classify(dataFrame, CValue, kernelToUse, testValuePercent, False))
            
        return accuracyResults

    #Collects the averages of C values in a range from 1 to the passed number of Cs to test
    @staticmethod
    def findCsUpToN(dataFrame, numCs, kernelToUse, testValuePercent, iterationsPerTest):
        cAverages = []
        cAverages.append(-1);
        for cTest in range(1, numCs+1):
            accuracyResults = SVM.testNIterations(dataFrame, cTest, kernelToUse, testValuePercent, iterationsPerTest)
            cAverages.append(SVM.findAverage(accuracyResults))
        return cAverages

    #Finds the average of a passed array
    @staticmethod
    def findAverage(resultArray):
        numResults = len(resultArray)
        return (float)(stats._sum(resultArray)[1]/numResults)

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

    @staticmethod
    def getValidationCurve(dataFrame):
        le = LabelEncoder()
        le.fit(dataFrame['Activity'].astype(str))

        y = le.transform(dataFrame['Activity'].astype(str))
        X = dataFrame.iloc[:, 2:102]

        param_range = np.logspace(-6, -1, 5)
        train_scores, test_scores = validation_curve(
            SVC(), X, y, param_name="gamma", param_range=param_range,
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
        plt.show()

    @staticmethod
    def getLearningCurve(dataFrame):
        le = LabelEncoder()
        le.fit(dataFrame['Activity'].astype(str))

        y = le.transform(dataFrame['Activity'].astype(str))
        X = dataFrame.iloc[:, 2:dataFrame.shape[1]]
        train_sizes = [1, 100, 250, 500, 750, 1000, 1250]

        train_scores, valid_scores = validation_curve(Ridge(), X, y, "alpha",
                                              np.logspace(-7, 3, 3),
                                              cv=5)

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

        plt.show()