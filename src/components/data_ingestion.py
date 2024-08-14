import os
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")

class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Starting the data ingestion process")
        try:
            # Load dataset
            df = pd.read_csv('notebook/data/stud.csv')
            logging.info('Dataset loaded successfully')

            # Ensure directory exists for saving datasets
            os.makedirs(os.path.dirname(self.config.train_data_path), exist_ok=True)

            # Save raw data
            df.to_csv(self.config.raw_data_path, index=False, header=True)
            logging.info("Raw data saved")

            # Train-test split
            logging.info("Initiating train-test split")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save train and test datasets
            train_set.to_csv(self.config.train_data_path, index=False, header=True)
            test_set.to_csv(self.config.test_data_path, index=False, header=True)
            logging.info("Data ingestion completed successfully")

            return self.config.train_data_path, self.config.test_data_path

        except Exception as e:
            logging.error("An error occurred during data ingestion")
            raise CustomException(e, sys)

if __name__ == "__main__":
    ingestion_obj = DataIngestion()
    ingestion_obj.initiate_data_ingestion()