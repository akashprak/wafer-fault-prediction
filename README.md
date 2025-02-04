# Wafer fault prediction - classification
A wafer is a thin slice of semiconductor.  
The goal is to build a machine learning model that can predict whether a given wafer is good or bad based on the values of its 590 sensors.  
The training data consists of sensor data classified as good or bad.
(+1) is used to denote a good wafer which is working.
(-1) is used to denote a bad wafer which is not.

### Schema
The data used for training and the data to be given for prediction should follow the structure of the schema file.  
The training data consists of a __Wafer id__ column, __590 columns of sensor values__ and the target columns which classifies the wafer as __Good/Bad__.  
The data is preprocessed and fed to the model for prediction.

## Architecture:
### Data Validation
The dataset is validated for the following:
- extension == .csv
- correct number of columns
- column names are as per the schema
The validated datasets are then merged into a single dataframe and saved to csv file and to a MySQL database.

### Database
A MySQL database is used to store and retrieve the training data using `sqlalchemy` library and `pymysql`.

### EDA
EDA is done on the merged dataset.  
- sensor-columns with more than 70% null values is dropped
- handled outliers
- certain sensor values are constant for the entire training data, which is dropped.

### Data Transformation
The training data is highly imbalanced. The (-1) values dominate the dataset.  
The dataset is balanced using `SMOTE` oversampler in the `imblearn` library.  
PCA is done on the balanced dataset to reduce the dimensionality.  
Note: Even after oversampling the trained model is slightly overfitted since the Good wafer (1) group has very small number of values to be representative of the underlying classification.

### Clustering
Clustering was tried to find any underlying clusters, but the results was against the existence of any.

### Model training
Four different models, __Logistic regression__, __SVC__, __Random forest__ and __XGBoost Classifier__ were trained on the transformed dataset.  
Hyperparameter tuning for the models was done using `GridSearchCV`.  
_f1 score_ was used as the scoring parameter so that more importance was given to the minority (1) class.   
Best model was based on the highest cross validated _f1 score_ and XGBClassifier was the model with the highest score after cross validation.  

## Steps:
### Installing required libraries
Necessary step for relative imports to work properly, using setup.py
```
pip install -r requirements.txt
```

### Training model
```
python src/training_pipeline.py
```

### Flask app
```
python app.py
```

### Running Docker on local
```
docker build -t waferdefaultprediction:latest .
docker run -p 7000:7000 waferdefaultprediction
```