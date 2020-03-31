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
Phone/Shoe
  Activity (Running, Driving, etc.)
    Data Source (Shoe 1, Shoe 2, acc, gyro)
      Event (Running Example 1, 2, etc.)
        Chunk
          Dataframes representing a portion of the .csv
        
The features are generated in the following scheme:
Phone/Shoe
  Activity (Running, Driving, etc.)
    Data Source (Shoe 1, Shoe 2, acc, gyro)
      Event (Running Example 1, 2, etc.)
        Dataframes representing the features generated for each chunk of data
        
The methods are relatively robust, so the program can handle different structures of data being passed in with the caveat that the FeatureUtil method that best reflects the point in the data being passed in must be used. 

1. Broadly speaking, the workflow is something like this:
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

### Overview of the Python Files:

- Classifier: The Driver class that contains main and some static methods for building the organizational data structure that will allow one to iterate over all of the data files in the project. 
- DataUtil: A static class that contains the methods to clean the data, convert between data types, and scale columns in the data.
- FeatureUtil: A static class that contains the methods for generating the features and analyzing them
- FileReaderUtil: A static class that contains methods to read in the filepaths of a given directory and read in an individual file.
- GraphUtil: An instance class that contains the methods used to create different graphs of the read-in data
- StringUtil: A static class that contains some additional string methods 
- SynchronizationUtil: A static class that contains the methods for breaking the data into chunks and synchronizing the data according to the first column (Time).



