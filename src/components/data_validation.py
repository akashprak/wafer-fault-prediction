import json
import shutil
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from src.database import conn_engine
from src.utils import merge_files


@dataclass
class DataValidationConfig:
    rawFiles_path = Path.cwd().joinpath("Training_Batch_Files")
    schema_path = Path("artifacts", "schema.json")
    validFiles = Path.cwd().joinpath("Valid_files")
    invalidFiles = Path.cwd().joinpath("Invalid_files")
    train_data_file_path = Path("data", "Merged_data.csv")

class DataValidation:
    """ This class handles all the validation done on the raw training data and the merging
    of the training batch files 
    """
    def __init__(self) -> None:
        self.data_validation_config = DataValidationConfig()

    def getSchemaValues(self) -> None:
        """
        This method extracts the relevant information about the valid training data from the schema file.

        Raise KeyError, ValueError, Exception on failure.
        """
        try:
            with open(self.data_validation_config.schema_path) as f:
                schema = json.load(f)

            self.lengthOfDateStamp = schema['LengthOfDateStamp']
            self.lengthOfTimeStamp = schema['LengthOfTimeStamp']
            self.Columns = schema['Columns']

            message = f'''
            LengthOfDateStamp: {self.lengthOfDateStamp}
            LengthOfTimeStamp: {self.lengthOfTimeStamp} 
            Number of Columns: {len(self.Columns)}'''
            logging.info(f"getting values from the schema:{message}")

        except KeyError:
            logging.error("Incorrect key passed to schema")
            raise KeyError
        
        except ValueError:
            logging.error(f"Value not found inside {self.data_validation_config.schema_path}")

        except Exception as e:
            logging.exception('Exception occured while getting schema values')
            raise CustomException(e)
        

    def validate_extension(self, filepath:Path) -> bool:
        """returns True if csv, else False"""
        try:
            extension = filepath.suffix
            return extension.lower()==".csv"
        
        except AttributeError:
            logging.error("function: validate_extension(), the filepath parameter should be of type 'pathlib.Path' ")
            raise AttributeError
    
    def validate_filename(self, filepath:Path) -> bool:
        """
        This method validates the filename for the correct format
        """
        try:
            name = filepath.stem    # gets the filename
            name = name.split("_")
            return ( (name[0].lower()=='wafer') and len(name[1])==self.lengthOfDateStamp and 
                len(name[2])==self.lengthOfTimeStamp )
        
        except AttributeError:
            logging.error("function: validate_filename(), the filepath parameter should be of type 'pathlib.Path' ")
            raise AttributeError

    def validate_columns(self, filepath:Path) -> bool:
        """validates the feature and target column names"""
        try:
            df = pd.read_csv(filepath)
            if len(df.columns[1:])!=len(self.Columns):
                logging.info(f"insufficient number of columns in the passed dataframe:  {filepath}")
                return False
            return (df.columns[1:]==self.Columns).all()
        
        except (FileNotFoundError, TypeError):
            logging.error(f"function: validate_columns(), the filepath parameter:{filepath} should be Path like")
            raise Exception

    def initiate_training_data_validation(self):
        try:
            self.getSchemaValues()

            self.data_validation_config.validFiles.mkdir(parents=True, exist_ok=True)
            self.data_validation_config.invalidFiles.mkdir(parents=True, exist_ok=True)
            logging.info("filename validation started")

            for filepath in self.data_validation_config.rawFiles_path.iterdir():
                
                if (
                    self.validate_extension(filepath) and 
                    self.validate_filename(filepath) and
                    self.validate_columns(filepath)
                ):
                    shutil.copy(filepath, self.data_validation_config.validFiles)
                    logging.info(f"filename: {filepath.name} is valid, copied to Valid_files")
                
                else:
                    shutil.copy(filepath, self.data_validation_config.invalidFiles)
                    logging.info(f"filename: {filepath.name} is Invalid, copied to Invalid_files")

            logging.info("merging valid Training_batch_files..")
            merged_df:pd.DataFrame = merge_files(source_dir=self.data_validation_config.validFiles)
            logging.info("merging successfull")

            parent_dir = self.data_validation_config.train_data_file_path.parent
            parent_dir.mkdir(parents=True, exist_ok=True)
            merged_df.to_csv(self.data_validation_config.train_data_file_path, index=False)
            logging.info(f"merged file saved in {self.data_validation_config.train_data_file_path}")

            try:
                merged_df.to_sql(name="merged_data", con=conn_engine, if_exists='replace', index=False)
                logging.info("Merged File written to database")
            
            except Exception as e:
                logging.exception("Exception while writing to SQL database")
        
        except Exception as e:
            logging.exception("function: initiate_training_data_validation() failed")
            raise CustomException(e)