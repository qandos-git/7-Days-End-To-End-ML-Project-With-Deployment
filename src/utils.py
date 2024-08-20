import os
import sys
import numpy as np 
import pandas as pd
import dill

from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path) #Extract directory name from file_path

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj: #with: Ensures file is properly opened and automatically closed after the code inside the with block is executed
            dill.dump(obj, file_obj) #Serializes (converts) the Python object obj into a binary format and writes it to the file specified
        '''
        dill is a Python module similar to pickle but with more flexibility, particularly in handling complex Python objects.
        '''
        logging.info("Saved successfully.")

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        report = {}

        for i, (model_name, model) in enumerate(models.items()):
            model.fit(X_train, y_train) 

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)