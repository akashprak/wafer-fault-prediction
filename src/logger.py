import logging 
from pathlib import Path
from datetime import datetime

LOG_FILE=f"{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log"
logs_path = Path.cwd().joinpath("logs")
logs_path.mkdir(parents=True, exist_ok=True)

LOG_FILE_PATH = logs_path.joinpath(LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)