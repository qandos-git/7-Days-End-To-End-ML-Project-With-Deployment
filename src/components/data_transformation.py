import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def _get_numerical_pipeline(self):
        """Creates a pipeline for numerical features."""
        return Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")), #Fill missing values with median
            ("scaler", StandardScaler()) #standrize neumirical values to closed range
        ])

    def _get_categorical_pipeline(self):
        """Creates a pipeline for categorical features."""
        return Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot_encoder", OneHotEncoder()), #[1,0,0]
            ("scaler", StandardScaler(with_mean=False))#[2,8, 0, 0]
            #with_mean=False -> data will not centred around zero (will not subtract the mean from each value)
        ])

    def get_data_transformer_object(self):
        """Creates a preprocessor object to transform numerical and categorical data."""
        '''
        - ColumnTransformer is used to apply different preprocessing steps to different columns in a dataset and then combine the results into a single feature matrix.
        - Pipelines are used within ColumnTransformer to define sequences of transformations for different subsets of columns.
        - This approach is preferable to using a single pipeline because it allows for tailored preprocessing of different data types, resulting in cleaner and more maintainable code.
        '''
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender", "race_ethnicity", "parental_level_of_education",
                "lunch", "test_preparation_course"
            ]

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")
            # apply different preprocessing pipelines to different 
            #
            preprocessor = ColumnTransformer(transformers=[
                ("num_pipeline", self._get_numerical_pipeline(), numerical_columns),
                ("cat_pipeline", self._get_categorical_pipeline(), categorical_columns)
            ])

            return preprocessor #matrix
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path: str, test_path: str):
        """Applies data transformation to training and test datasets."""
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info('Reading train and test data completed')

            preprocessing_obj = self.get_data_transformer_object()
            target_column = 'math_score'

            input_features_train_df = train_df.drop(columns=[target_column])
            target_feature_train_df = train_df[target_column]

            input_features_test_df = test_df.drop(columns=[target_column])
            target_feature_test_df = test_df[target_column]

            logging.info("Applying preprocessing object on training and testing data.")
            ''''
            fit_transform use (fit) once in training_data to compute data statistics (mean/median..), then apply transformations using those statistics (transform). 
            transform used on testing data and any new data to apply the same transformations from the training data without recalculating statistics (mean/median..), ensuring consistency.
            This approach prevents data leakage and maintains model integrity during training and evaluation.
            '''
            
            input_feature_train_arr = preprocessing_obj.fit_transform(input_features_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_features_test_df)

            '''
            C_: Concatenate arrays by columns.
            '''
            train_arr = np.c_[input_feature_train_arr, target_feature_train_df.to_numpy()]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_df.to_numpy()]

            logging.info("Saving preprocessing object.")
            save_object(file_path=self.config.preprocessor_obj_file_path, obj=preprocessing_obj)

            return train_arr, test_arr, self.config.preprocessor_obj_file_path
        except Exception as e:
            raise CustomException(e, sys)
