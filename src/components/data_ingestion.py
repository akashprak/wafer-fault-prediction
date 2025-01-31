import pandas as pd
from pathlib import Path
from dataclasses import dataclass

from src.database import conn_engine
from src.logger import logging
from src.exception import CustomException

@dataclass
class DataIngestionConfig:
    train_data_path = Path("artifacts", "train.csv")

class DataIngestion:
    """
    This class handles the reading of the training data from the database.
    """

    def __init__(self):
        self._ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self) -> Path:
        """
        Returns: the train data path of the ingested csv file
        """
        try:
            logging.info("Data ingestion starts")

            try:
                df = pd.read_sql_table(table_name="cleaned_data", con=conn_engine)
                logging.info("Dataset read from the database")

            except Exception as e:
                logging.exception("Failed to read from the database")
                raise CustomException(e)
            
            df.to_csv(self._ingestion_config.train_data_path)
            logging.info("Data Ingestion completed")

            return self._ingestion_config.train_data_path
        
        except Exception as e:
            logging.exception("initiate_data_ingestion failed")
            raise CustomException(e)