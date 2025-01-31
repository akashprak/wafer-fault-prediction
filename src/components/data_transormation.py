import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from imblearn.over_sampling import SMOTE
from pathlib import Path
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    prediction_prepocessor_obj_filepath = Path("artifacts", "prediction_preprocessor.pkl")
    feature_names_path = Path("artifacts", "valid_features.txt")

class DataTransformation:
    """
    This class handles the data transformation on the valid data fetched from the raw training data
    """

    scaler = RobustScaler()
    imputer = KNNImputer()
    oversampler = SMOTE(sampling_strategy=0.33, random_state=10)
    pca = PCA(n_components=0.95, random_state=55)
    labelenc = LabelEncoder()

    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def _training_preprocessor(self, X_train, y_train) -> tuple[np.ndarray, np.ndarray]:
        """
        This method handles the preprocessing of the training data
        Returns: X_train, y_train
        """
        try:
            logging.info("Preprocessing of the training data started")

            # label encoding the target to be consistent with the metrics and
            # xgboost require target labels to be encoded from 0
            y_train = self.labelenc.fit_transform(y_train)

            # the preprocessing steps wil be done one by one since oversampling needs to be done in between
            # and the oversampled y values need to be returned

            X_train = self.scaler.fit_transform(X_train)
            X_train = self.imputer.fit_transform(X_train)

            X_train , y_train = self.oversampler.fit_resample(X_train, y_train)     # oversampling
            
            X_train = self.pca.fit_transform(X_train)

            logging.info("Training data successfully preprocessed")
            logging.info(f"shape of data after preprocessing: {X_train.shape}")
            return X_train, y_train

        except Exception as e:
            logging.exception("exception while training preprocessing")
            raise CustomException(e)
        
    def _prediction_preprocessor(self):
        """
        This method returns the preprocessor pipeline to be used for prediction
        Returns: preprocessor Pipeline object
        """

        try:
            logging.info("Creating preprocessor pipeline")
            preprocessor = Pipeline(
                steps=[
                    ("scaler", self.scaler),
                    ("imputer", self.imputer),
                    ("pca", self.pca)
                ]
            )
            return preprocessor
        
        except Exception as e:
            logging.exception("Exception while creating preprocessor pipeline for prediction data")


    def initiate_data_transformation(self, train_path) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns: X_train_arr , y_train_arr"""
        try:
            df = pd.read_csv(train_path)
            logging.info("Training data read from csv")
            logging.info(f"Training dataframe sample: \n{df.iloc[:5, :10]}")
            
            try:
                with open(self.transformation_config.feature_names_path) as file:
                    valid_features = file.readlines()
                valid_features = [line.strip() for line in valid_features]
                target = "Good/Bad"
            
            except Exception as e:
                logging.exception("Exception while reading valid_features.txt")
                raise CustomException(e)

            feature_df = df[valid_features]
            target_df = df[target]

            logging.info("preprocessing the training dataset..")
            feature_df, target_df = self._training_preprocessor(feature_df, target_df)

            logging.info("getting feature_preprocessor object")
            preprocessor = self._prediction_preprocessor()

            save_object(
                file_path = self.transformation_config.prediction_prepocessor_obj_filepath,
                obj = preprocessor
            )
            logging.info("preprocessor saved as pickle file")

            X_train = np.array(feature_df)
            y_train = np.array(target_df)

            return X_train, y_train

        except Exception as e:
            logging.exception("Exception while initiating data transformation")
            raise CustomException(e)