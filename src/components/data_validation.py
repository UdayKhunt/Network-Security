from src.exception import NetworkSecurityException
from src.logger import logging
from src.entity.config_entity import DataValidationConfig
from src.entity.artifact_entity import DataValidationArtifact , DataIngestionArtifact
from src.constants.training_pipeline import DATA_SCHEMA_PATH
from src.utils.main_utils import read_yaml_file , write_yaml_file
import os,sys
import pandas as pd
from scipy.stats import ks_2samp

class DataValidation:
    def __init__(self , data_validation_config : DataValidationConfig , data_ingestion_artifact : DataIngestionArtifact):
        try:
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.schema_config = read_yaml_file(DATA_SCHEMA_PATH)
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    def validate_number_of_columns(self , df : pd.DataFrame):
        try:
            num1 = len(self.schema_config)
            num2 = len(list(df.columns))

            return num1 == num2
        
        except Exception as e:
            raise NetworkSecurityException(e,sys)


    def detect_dataset_drift(self , base_df , current_df , threshold = .05):
        try:
            report = {}
            status = True
            for column in base_df.columns:
                d1 = base_df[column]
                d2 = current_df[column]
                is_dist_same = ks_2samp(d1 , d2)

                if is_dist_same.pvalue >= threshold:
                    isFound = False
                else:
                    isFound = True
                    status = False
                report.update({column : {
                    'p value' : float(is_dist_same.pvalue),
                    'drift status' : isFound
                }})
            
            drift_report_file_path = self.data_validation_config.drift_report
            os.makedirs(os.path.dirname(drift_report_file_path))
            write_yaml_file(drift_report_file_path , report)            

            return status
        
        except Exception as e:
            raise NetworkSecurityException(e,sys)


    def initiate_data_validation(self):
        try:

            train_data_path = self.data_ingestion_artifact.train_file_path
            test_data_path = self.data_ingestion_artifact.test_file_path

            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)

            status = self.validate_number_of_columns(train_df)
            if not status:
                error_message = "Train data does not contain all columns"
            status = self.validate_number_of_columns(test_df)
            if not status:
                error_message = "Test data does not contain all columns"

            status = self.detect_dataset_drift(train_df , test_df)

            #whether status is true or false, it doesnt matter, we will store train and test data in their respective file paths.
            # To check status, we can directly see it from the data validationa artifact and proceed accordingly

            os.makedirs(os.path.dirname(self.data_validation_config.valid_train_file_path))

            train_df.to_csv(self.data_validation_config.valid_train_file_path , index = False , header = True)
            test_df.to_csv(self.data_validation_config.valid_test_file_path , index = False , header = True)

            data_validation_artifact = DataValidationArtifact(
                status , 
                self.data_validation_config.valid_train_file_path,
                self.data_validation_config.valid_test_file_path,
                None,
                None,
                self.data_validation_config.drift_report
            )

            return data_validation_artifact
        
        except Exception as e:
            raise NetworkSecurityException(e,sys)


        


    

