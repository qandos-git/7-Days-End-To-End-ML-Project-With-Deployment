import logging
import os
from datetime import datetime

# Define log file name with timestamp
log_file_name = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Define the directory and file path for logs
logs_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_dir, exist_ok=True) #it is ok if the dir exist, don't raise error, use the exist dir

log_file_path = os.path.join(logs_dir, log_file_name)

# Configure logging
logging.basicConfig(
    filename=log_file_path,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
 