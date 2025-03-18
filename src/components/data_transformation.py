from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataValidationArtifact , DataTransformationArtifact
from src.exception import NetworkSecurityException
from src.logger import logging
import os,sys
import pandas as pd
from src.constants.training_pipeline import TARGET_COLUMN , DATA_TRANSFORMATION_PROCESSOR_PARAMETERS
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
import numpy as np
from src.utils.main_utils import save_numpy_array , save_obj

class DataTransformation:
    def __init__(self , data_transformation_config :  DataTransformationConfig, data_validation_artifact : DataValidationArtifact):
        try: 
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact

        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    def get_preprocessor_obj(cls):
        try:
            imputer = KNNImputer(**DATA_TRANSFORMATION_PROCESSOR_PARAMETERS)
            processor = Pipeline([('knn_imputer' , imputer)])
            return processor
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    def initiate_data_transformation(self):
        try:
            train_data = pd.read_csv(self.data_validation_artifact.valid_train_filepath)
            test_data = pd.read_csv(self.data_validation_artifact.valid_test_filepath)

            input_features_train_data = train_data.drop(TARGET_COLUMN , axis = 1)
            output_train_data = train_data[TARGET_COLUMN]
            output_train_data = output_train_data.replace(-1,0)

            input_features_test_data = test_data.drop(TARGET_COLUMN , axis = 1)
            output_test_data = test_data[TARGET_COLUMN]
            output_test_data = output_test_data.replace(-1,0)

            preprocessor = self.get_preprocessor_obj()
            preprocessor_obj = preprocessor.fit(input_features_train_data)
            transformed_input_features_train_data = preprocessor_obj.transform(input_features_train_data)
            transformed_input_features_test_data = preprocessor_obj.transform(input_features_test_data)

            train_arr = np.c_[transformed_input_features_train_data , np.array(output_train_data)]
            test_arr = np.c_[transformed_input_features_test_data , np.array(output_test_data)]

            save_numpy_array(self.data_transformation_config.transformed_train_path , train_arr)
            save_numpy_array(self.data_transformation_config.transformed_test_path , test_arr)

            save_obj(self.data_transformation_config.transformed_obj_path , preprocessor_obj)

            data_transformation_artifact = DataTransformationArtifact(self.data_transformation_config.transformed_train_path,
                                                                      self.data_transformation_config.transformed_test_path,
                                                                      self.data_transformation_config.transformed_obj_path)

            return data_transformation_artifact            
        
        except Exception as e:
            raise NetworkSecurityException(e,sys)
          

