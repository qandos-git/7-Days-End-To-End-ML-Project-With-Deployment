import os
import sys
from dataclasses import dataclass
from typing import Dict

from catboost import CatBoostRegressor
import numpy as np
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array: np.ndarray, test_array: np.ndarray) -> float:
        """Train models, evaluate them, and save the best one."""
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            # Define models
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors": KNeighborsRegressor(),
                "XGBoost": XGBRegressor(),
                "CatBoost": CatBoostRegressor(verbose=False),
                "AdaBoost": AdaBoostRegressor(),
            }

            # Evaluate models
            model_report = evaluate_models(X_train, y_train, X_test, y_test, models)

            # Find the best model name and score using max() with a key argument
            best_model_name, best_model_score = max(model_report.items(), key=lambda item: item[1])
            best_model = models[best_model_name]

            # Check if the best model score is satisfactory
            if best_model_score < 0.6:
                raise CustomException("No suitable model found with a score above 0.6")

            logging.info(f"Best model found: {best_model_name} with score: {best_model_score}")

            # Save the best model
            save_object(file_path=self.config.trained_model_file_path, obj=best_model)

            # Predict and calculate r2_score
            predictions = best_model.predict(X_test)
            r2_square = r2_score(y_test, predictions)

            return r2_square

        except Exception as e:
            raise CustomException(e, sys)
