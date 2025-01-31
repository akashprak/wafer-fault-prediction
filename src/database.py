from sqlalchemy import create_engine,URL
from src.exception import CustomException
from src.logger import logging

_url_obj = URL.create(
    drivername="mysql+pymysql",
    username="root",
    password="akash",
    host="localhost",
    port=3306,
    database='wafer_data'
)

def _createEngine():
    try:
        engine = create_engine(_url_obj)
        return engine
    
    except Exception as e:
        logging.exception("Exception during SQL engine creation")
        CustomException(e)

conn_engine = _createEngine()