from src.constants import training_pipeline
from datetime import datetime
import os

class TrainingPipelineConfig:
    def __init__(self , timestamp = datetime.now()):
        timestamp=timestamp.strftime("%m_%d_%Y_%H_%M_%S")
        self.target_column = training_pipeline.TARGET_COLUMN
        self.pipeline_name = training_pipeline.PIPELINE_NAME
        self.artifact_dir_name = training_pipeline.ARTIFACT_DIR
        self.artifact_path = os.path.join(self.artifact_dir_name , timestamp)
        self.timestamp = timestamp
        '''
        TARGET_COLUMN = 'Result'
        PIPELINE_NAME = 'NetworkSecurity'
        ARTIFACT_DIR = 'artifact'
        DATA_FILE_NAME = 'phisingData.csv'
        '''

class DataIngestionConfig:
    def __init__(self , training_pipeline_config : TrainingPipelineConfig):
        self.data_ingestion_dir = os.path.join(training_pipeline_config.artifact_path , 
                                               training_pipeline.DATA_INGESTION_DIR_NAME)
        self.feature_store_dir = os.path.join(self.data_ingestion_dir , 
                                              training_pipeline.DATA_INGESTION_FEATURE_STORE , training_pipeline.DATA_FILE_NAME)
        self.training_file_path = os.path.join(self.data_ingestion_dir , 
                                              training_pipeline.DATA_INGESTION_INGESTED , training_pipeline.TRAIN_FILE_NAME)
        self.testing_file_path = os.path.join(self.data_ingestion_dir , 
                                              training_pipeline.DATA_INGESTION_INGESTED , training_pipeline.TEST_FILE_NAME)
        self.train_test_split_ratio = training_pipeline.DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
        self.collection_name = training_pipeline.COLLECTION_NAME
        self.database_name = training_pipeline.DATABASE_NAME


class DataValidationConfig:
    def __init__(self , training_pipeline_config : TrainingPipelineConfig):
        self.data_validation_dir = os.path.join(training_pipeline_config.artifact_path , training_pipeline.DATA_VALIDATION_DIR_NAME)
        self.valid_data_dir = os.path.join(self.data_validation_dir , training_pipeline.DATA_VALIDATION_VALID_DATA_DIR_NAME)
        self.invalid_data_dir = os.path.join(self.data_validation_dir , training_pipeline.DATA_VALIDATION_INVALID_DATA_DIR_NAME)
        self.valid_train_file_path = os.path.join(self.valid_data_dir , training_pipeline.TRAIN_FILE_NAME)
        self.valid_test_file_path = os.path.join(self.valid_data_dir , training_pipeline.TEST_FILE_NAME)
        self.invalid_train_file_path = os.path.join(self.invalid_data_dir , training_pipeline.TRAIN_FILE_NAME)
        self.invalid_test_file_path = os.path.join(self.invalid_data_dir , training_pipeline.TRAIN_FILE_NAME)
        self.drift_report = os.path.join(self.data_validation_dir ,
                                          training_pipeline.DATA_VALIDATION_DRIFT_REPORT_DIR_NAME , 
                                          training_pipeline.DATA_VALIDATION_DRIFT_REPORT_FILE_NAME
        )

        """

        DATA_VALIDATION_DIR_NAME = 'data_validation'
        DATA_VALIDATION_VALID_DATA_DIR_NAME = 'valid'
        DATA_VALIDATION_INVALID_DATA_DIR_NAME = 'invalid'
        DATA_VALIDATION_DRIFT_REPORT_DIR_NAME = 'drift_report'
        DATA_VALIDATION_DRIFT_REPORT_FILE_NAME = 'report.yaml'

        """

class DataTransformationConfig:
    def __init__(self , trainig_pipeline_config : TrainingPipelineConfig):
        self.data_transformation_dir = os.path.join(trainig_pipeline_config.artifact_path , training_pipeline.DATA_TRANSFORMATION_DIR)
        self.transformed_train_path = os.path.join(self.data_transformation_dir , 
        training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR , training_pipeline.TRAIN_FILE_NUMPY_PATH)
        self.transformed_test_path = os.path.join(self.data_transformation_dir , 
        training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR , training_pipeline.TEST_FILE_NUMPY_PATH)
        self.transformed_obj_path = os.path.join(self.data_transformation_dir , 
        training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR , training_pipeline.PROCESSING_OBJ_FILE_NAME)

class ModelTrainerConfig:
    def __init__(self , trainig_pipeline_config : TrainingPipelineConfig):
        self.model_trainer_dir = os.path.join(trainig_pipeline_config.artifact_path , 
                                              training_pipeline.MODEL_TRAINER_DIR)
        self.trained_model_file_path = os.path.join(self.model_trainer_dir , training_pipeline.MODEL_TRAINER_TRAINED_MODEL_DIR ,
                                                    training_pipeline.MODEL_TRAINER_TRAINED_MODEL_NAME)
        self.expected_accuracy = training_pipeline.MODEL_TRAINER_EXPECTED_ACCURACY
        self.overfitting_underfitting_threshold = training_pipeline.MODEL_TRAINER_OVER_FIITING_UNDER_FITTING_THRESHOLD
