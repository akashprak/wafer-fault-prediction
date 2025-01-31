import pickle
import pandas as pd
from pathlib import Path
from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    """
    for saving objects to the given path
    """
    try:
        dir_path = Path(file_path).parent
        dir_path.mkdir(parents=True, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        logging.exception("Exception in the save_object util")
        raise CustomException(e)
    

def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
        
    except Exception as e:
        logging.error('Exception Occured in load_object function utils')
        raise CustomException(e)
    
def merge_files(source_dir:Path) -> pd.DataFrame :
    """merges the csv files in source_dir into a single file"""
    try:
        files = source_dir.glob("*.csv")
        logging.info(f"merging csv files in {source_dir}")
        df = pd.concat(map(pd.read_csv, files), ignore_index=True)
        return df

    except (FileNotFoundError, TypeError):
        logging.error("the source_dir parameter should be of type pathlib.Path")
        raise Exception
        
    except Exception as e:
        logging.error("merging the csv files failed")
        raise CustomException(e)

def clear_directory(dir:Path):
    """clear files inside a directory"""
    try:
        for file in dir.iterdir():
            if file.is_file():
                file.unlink()
            elif file.is_dir():
                file.rmdir()
    except Exception as e:
        logging.exception(f"error while clearing {dir}")
        raise CustomException(e)