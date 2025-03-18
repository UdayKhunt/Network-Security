import os
import numpy as np

#Training Pipeline Config
TARGET_COLUMN = 'Result'
PIPELINE_NAME = 'NetworkSecurity'
ARTIFACT_DIR = 'artifact'
DATA_FILE_NAME = 'phisingData.csv'

#Data Ingestion Config
DATABASE_NAME = 'Uday'
COLLECTION_NAME = 'NetworkSecurityData'
DATA_INGESTION_DIR_NAME = 'data_ingestion'
DATA_INGESTION_FEATURE_STORE = 'feature_store'
DATA_INGESTION_INGESTED = 'ingested'
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO = .2
TRAIN_FILE_NAME = 'train.csv'
TEST_FILE_NAME = 'test.csv'

#Data Validation Config
DATA_VALIDATION_DIR_NAME = 'data_validation'
DATA_VALIDATION_VALID_DATA_DIR_NAME = 'valid'
DATA_VALIDATION_INVALID_DATA_DIR_NAME = 'invalid'
DATA_VALIDATION_DRIFT_REPORT_DIR_NAME = 'drift_report'
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME = 'report.yaml'
DATA_SCHEMA_PATH = os.path.join('data_schema' , 'schema.yaml')


#Data Transformation Config
DATA_TRANSFORMATION_DIR = 'data_transformation'
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR = 'transformed'
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR = 'transformed_object'
TRAIN_FILE_NUMPY_PATH = 'train.npy'
TEST_FILE_NUMPY_PATH = 'test.npy'
PROCESSING_OBJ_FILE_NAME = 'preprocessor.pkl'

DATA_TRANSFORMATION_PROCESSOR_PARAMETERS = {
    'missing_values' : np.nan,
    'n_neighbors' : 3,
    'weights' : 'uniform'
}

#Model Trainer Config
MODEL_TRAINER_DIR = 'model_trainer'
MODEL_TRAINER_TRAINED_MODEL_DIR = 'trained_model'
MODEL_TRAINER_TRAINED_MODEL_NAME = 'model.pkl'
MODEL_TRAINER_EXPECTED_ACCURACY = .6
MODEL_TRAINER_OVER_FIITING_UNDER_FITTING_THRESHOLD = .05

