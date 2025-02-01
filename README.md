## Wafer fault prediction - classification
A wafer is a thin slice of semiconductor.  
The goal is to build a machine learning model that can predict whether a given wafer is good or bad based on the values of its 590 sensors.  
The training data consists of sensor data classified as good or bad.
(+1) is used to denote a good wafer which is working.
(-1) is used to denote a bad wafer which is not.

### Schema
The data used for training and the data to be given for prediction should follow the structure of the schema file.  
The training data consists of a __Wafer id__ column, __590 columns of sensor values__ and the target columns which classifies the wafer as __Good/Bad__.  
The data is preprocessed and fed to the model for prediction.

### Architecture:
### Data Validation
The dataset is validated for the following:
- extension == .csv
- correct number of columns
- column names are as per the schema
The validated datasets are then merged into a single dataframe and saved to csv file and to a MySQL database.
### Database
A MySQL database is used to store and retrieve the training data using `sqlalchemy` library.