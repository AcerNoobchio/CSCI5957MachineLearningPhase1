# Machine Learning Activity Classification Phase 1

## Authors: Koi Stephanos and Jacob Hoyos

### General Project Details:

- Purpose: To prepare a dataset for use with machine learning algorithms by cleaning the data, dividing it, and selecting features using - feature analysis.
- Language: Python v3.7
- Status: Building and Running - Last Updated 3/31/2020
- Included Libraries: Pandas, Numpy, Statistics, SciPy
- Current Issues: None

### Installation:

The project was originally created using cookiecutter code from visual studio, so, for the most seamless experience, using Visual Studio with the python development environment is highly recommended. Otherwise, the directory contains several python files that can be run. The Classifier file contains main, so if run from the commandline, the Classifier file should be the one to be executed. 

### Usage Information:

Main is already structured for use. Simply uncomment the needed lines and execute the program. The important thing to note is that the filepaths for the directory, outputDirectory, and featureDirectory variables must be changed to reflect one's own file structure. 
directory: This is the path of the directory to read the data from.
outputDirectory: This is the path where one would like graphs to be saved to.
featureDirectory: This is the path of the directory where one would like to export feature data to.

It is extremely important that the filesystem for the data matches the file system that is in the repository! These methods are robust, but they are not designed to handle all structures of data, so some thought is advised before attempting to use different file schemes to produce and rank features.

The data is processed and cleaned in the following scheme:
  - Phone/Shoe
  - Activity (Running, Driving, etc.)
  - Source (Shoe 1, Shoe 2, acc, gyro)
  - (Running Example 1, 2, etc.)
  - Chunk
  - Dataframes representing a portion of the .csv
        
The features are generated in the following scheme:
  - Phone/Shoe
  - Activity (Running, Driving, etc.)
  - Data Source (Shoe 1, Shoe 2, acc, gyro)
  - Event (Running Example 1, 2, etc.)
  - Dataframes representing the features generated for each chunk of data
        
The methods are relatively robust, so the program can handle different structures of data being passed in with the caveat that the FeatureUtil method that best reflects the point in the data being passed in must be used. 

Broadly speaking, the workflow for preprocessing the data is something like this:
  - Read in the data
    - Use ReadFilePaths or ReadByFileRate in FileReaderUtil
  - Synchronize the data
  - Clean the data 
    - Both this step and the above step can be completed using the synchronizeDataFromPaths method above main
  - Graph the data (optional)
    - Use plotDirectory, plotFeature, or plotGraph in GraphUtil
  - Generate Features
    - Use exportDataSetFeatures in FeatureUtil
  - Analyze Features
    - Use rankFeatures in FeatureUtil
  - Graph the Feature Analysis (optional)
  
### Classifying Data:

Currently three machine learning algorithms have been implemented and are capable of providing predictions. These three models include:
- SVM
- Logistic Regression
- Multi Layer Perceptron (MLP)

In order to use the models, create an instance of the class within classifier.py. A variety of options such as test_split ratio, regularization rate and more can be selected to customize performance. Class methods for each model implementation also include support for printing a variety of metrics and performance graphs.

Exampe usage of SVM (params: dataFrame, CValue, kernelToUse, testSizePercent, isFixed):

**SVM.classify(allCombined, 1, 'linear', 20, True)**

Example usage of LogReg(params: dataFrame, testSizePercent, regularizationRate):

**LogReg.classify(allCombined, .2, .01)**

Example usage of Neural Network (params: dataFrame, alphaValue, layerDimensions, activationFunction, solver, testSizePercent, isFixed, printResults):

**NeuralNetwork.classify(allCombined, 0.083, (100,100), 'logistic', 'adam', 20, false, true)**

### Overview of the Python Files:

- Classifier: The Driver class that contains main and some static methods for building the organizational data structure that will allow one to iterate over all of the data files in the project
- Classifier Util: Contains the wrapper functions to do the data cleaning 
- DataUtil: A static class that contains the methods to clean the data, convert between data types, and scale columns in the data
- FeatureUtil: A static class that contains the methods for generating the features and analyzing them
- FileReaderUtil: A static class that contains methods to read in the filepaths of a given directory and read in an individual file
- GraphUtil: An instance class that contains the methods used to create different graphs of the read-in data
- LogisiticRegression: A static class that contains methods for testing and creating a logistic regression model
- NeuralNetwork: A class that contains methods for testing and creating a neural network model using a multi-layer perceptron
- StringUtil: A static class that contains some additional string methods 
- SupportVector: A static class that conatins methods for creating and testing a Support Vector Machine classification model
- SynchronizationUtil: A static class that contains the methods for breaking the data into chunks and synchronizing the data according to the first column (Time).

## License

This project is available as open source under the terms of the [MIT License](http://opensource.org/licenses/MIT).

