import shutil
import pandas as pd
from pathlib import Path
from src.logger import logging
from src.exception import CustomException
from src.utils import load_object, merge_files, clear_directory
from src.components.data_validation import DataValidation
from dataclasses import dataclass

@dataclass
class PredictionConfig:
    validated_data_filepath = Path("Prediction_data_validated")
    preprocessor_filepath = Path("artifacts", "prediction_preprocessor.pkl")
    model_filepath = Path("artifacts", "best_model.pkl")
    feature_names_path = Path("artifacts", "valid_features.txt")

class PredictionFileValidation:
    """This class handles the validation of the data given to the model for prediction"""

    def __init__(self):
        self.data_validation = DataValidation()
        self.prediction_config = PredictionConfig()

    def validate(self, source_dir:Path) -> pd.DataFrame:
        """
        validates the csv files inside the input directory, merges them and returns it as a single dataframe
        Returns: pandas Dataframe
        """
        try:
            self.data_validation.getSchemaValues()
            self.prediction_config.validated_data_filepath.mkdir(parents=True, exist_ok=True)
            
            logging.info("validating the prediction data")
            for file in source_dir.iterdir():
                if (
                    self.data_validation.validate_extension(file) and 
                    self.data_validation.validate_columns(file)
                    ):
                    shutil.copy(file, self.prediction_config.validated_data_filepath)

            if not any(self.prediction_config.validated_data_filepath.iterdir()):
                # returning empty dataframe
                return pd.DataFrame()
            
            prediction_df = merge_files(self.prediction_config.validated_data_filepath)
            logging.info(prediction_df.head().iloc[:,:5].to_string())
            return prediction_df
        
        except Exception as e:
            logging.error("validating the prediction data failed")
            raise CustomException(e)

        finally:
            clear_directory(self.prediction_config.validated_data_filepath)
            logging.info("clearing the Prediction_data_validated directory")


class PredictionPipeline:
    def __init__(self):
        self.prediction_config = PredictionConfig()

        self.preprocessor = load_object(self.prediction_config.preprocessor_filepath)
        self.model = load_object(self.prediction_config.model_filepath)

    def predict(self, dataframe:pd.DataFrame) -> pd.DataFrame:
        try:
            with open(self.prediction_config.feature_names_path) as file:
                valid_features = file.readlines()
            valid_features = [line.strip() for line in valid_features]
        except Exception as e:
            logging.error("Exception while reading valid_features.txt")
            raise CustomException(e)
        
        try:
            wafer_names = dataframe.iloc[:,0]
            dataframe = dataframe[valid_features]

            data_preprocessed = self.preprocessor.transform(dataframe)
            prediction = self.model.predict(data_preprocessed)
            mapper = {0:"Bad", 1:"Good"}
            # converting numeric predictions into string values
            prediction = pd.Series(prediction).map(mapper)
            # joining the result with the wafer ids
            prediction_df = pd.concat([wafer_names, prediction], axis=1)
            logging.info(f"predicted result:\n {(prediction_df.head().iloc[:,:5]).to_string()}")
            return prediction_df

        except Exception as e:
            logging.error("execption occured while transforming and predicting on the prediction data")
            raise CustomException(e)